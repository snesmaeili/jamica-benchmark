"""Provenance stamping in the comparator runners' shared write_result().

Hermetic: no venv, no network, and no AMICA compute path — the distribution
lookups are faked, so this runs identically under JAX and the NumPy fallback.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_common():
    path = Path(__file__).resolve().parents[1] / "benchmark" / "comparator" / "runners" / "_common.py"
    spec = importlib.util.spec_from_file_location("runner_common", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeMeta(dict):
    """Minimal stand-in for importlib.metadata metadata (name lookup only)."""


class _FakeDist:
    def __init__(self, name, version, direct_url=None):
        self.version = version
        self.metadata = _FakeMeta(Name=name)
        self._direct_url = direct_url

    def read_text(self, filename):
        if filename == "direct_url.json":
            return self._direct_url
        return None


def _install_fake_metadata(monkeypatch, common, dists):
    """Patch _common's importlib_metadata so probing resolves to `dists`.

    `dists` maps a PEP 503-normalized name -> _FakeDist. version()/distribution()
    normalize their argument the way importlib.metadata does, so probing both
    "pyamica" and "pyAMICA" hits the same entry.
    """
    def _norm(name):
        return name.lower().replace("_", "-").replace(".", "-")

    def fake_distribution(name):
        d = dists.get(_norm(name))
        if d is None:
            raise common.importlib_metadata.PackageNotFoundError(name)
        return d

    def fake_version(name):
        return fake_distribution(name).version

    monkeypatch.setattr(common.importlib_metadata, "distribution", fake_distribution)
    monkeypatch.setattr(common.importlib_metadata, "version", fake_version)


def test_git_install_records_url_and_commit(monkeypatch):
    common = load_common()
    sha = "8e5744b3c67ed98a6e9ff56e9745b5b0eca3e0da"
    direct_url = json.dumps({
        "url": "git+https://github.com/DerAndereJohannes/pyamica.git",
        "vcs_info": {"vcs": "git", "commit_id": sha},
    })
    _install_fake_metadata(monkeypatch, common, {
        "pyamica": _FakeDist("pyamica", "0.4.1", direct_url),
    })
    prov = common.stack_provenance()
    pkg = prov["packages"]["pyamica"]
    assert pkg["version"] == "0.4.1"
    assert pkg["commit"] == sha
    # The "git+" scheme prefix is stripped; the github.com/owner/repo survives.
    assert pkg["url"] == "https://github.com/DerAndereJohannes/pyamica.git"
    assert prov["python"] and prov["executable"]


def test_case_variant_probes_dedupe_to_one_entry(monkeypatch):
    """"pyamica" and "pyAMICA" normalize to one dist; report it once."""
    common = load_common()
    _install_fake_metadata(monkeypatch, common, {
        "pyamica": _FakeDist("pyamica", "0.4.1", None),
    })
    prov = common.stack_provenance("pyamica", "pyAMICA")
    assert list(prov["packages"]) == ["pyamica"]


def test_wheel_install_has_version_but_no_commit(monkeypatch):
    common = load_common()
    _install_fake_metadata(monkeypatch, common, {
        "amica": _FakeDist("amica", "1.2.3", None),  # no direct_url.json
    })
    prov = common.stack_provenance("amica")
    assert prov["packages"]["amica"] == {"version": "1.2.3"}


class _ExplodingMeta:
    def __getitem__(self, key):
        raise RuntimeError("malformed METADATA")


class _BadDist:
    version = "0.0.0"
    metadata = _ExplodingMeta()

    def read_text(self, filename):
        return None


def test_malformed_metadata_does_not_crash(monkeypatch):
    """A dist with unreadable metadata is skipped, never crashes a result write."""
    common = load_common()

    def fake_distribution(name):
        if common._canonical_dist_key(name) == "pyamica":
            return _BadDist()
        raise common.importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(common.importlib_metadata, "distribution", fake_distribution)
    monkeypatch.setattr(common.importlib_metadata, "version",
                        lambda n: (_ for _ in ()).throw(common.importlib_metadata.PackageNotFoundError(n)))
    prov = common.stack_provenance("pyamica")  # must not raise
    assert prov["packages"] == {}


def test_pep503_canonical_key_collapses_separators():
    common = load_common()
    k = common._canonical_dist_key
    # case-only differences collapse (the real pyamica vs pyAMICA collision)
    assert k("pyamica") == k("pyAMICA") == "pyamica"
    # runs of [-_.] collapse to a single '-' (PEP 503), so these are one name
    assert k("scikit_learn") == k("scikit-learn") == k("scikit__learn") == "scikit-learn"


def test_amica_src_overrides_amica_commit(monkeypatch):
    """Under AMICA_SRC the measured amica is the checkout, not the installed wheel:
    the amica entry's commit must be the checkout's git HEAD + commit_source."""
    common = load_common()
    _install_fake_metadata(monkeypatch, common, {
        "amica": _FakeDist("amica", "0.3.0", None),  # installed wheel, no vcs commit
    })
    monkeypatch.setenv("AMICA_SRC", "/scratch/x/amica-blocked")
    import subprocess

    class _R:
        returncode = 0
        stdout = "b" * 40 + "\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _R())
    prov = common.stack_provenance("amica")
    amica = prov["packages"]["amica"]
    assert amica["commit"] == "b" * 40
    assert amica["commit_source"] == "amica_src"
    assert amica["src"] == "/scratch/x/amica-blocked"


def test_pin_optout_flags_recorded(monkeypatch):
    common = load_common()
    _install_fake_metadata(monkeypatch, common, {})
    monkeypatch.setenv("AMICA_SKIP_PIN_CHECK", "1")
    prov = common.stack_provenance()
    assert prov["env_flags"]["AMICA_SKIP_PIN_CHECK"] == "1"


def test_write_result_injects_provenance(monkeypatch, tmp_path):
    common = load_common()
    _install_fake_metadata(monkeypatch, common, {})  # no impls installed
    out = tmp_path / "r.json"
    common.write_result(str(out), {
        "implementation": "stub", "fit_time_s": 0.1, "peak_rss_gb": 0.1,
        "ll_final": -1.0,
    })
    doc = json.loads(out.read_text())
    assert "provenance" in doc
    assert set(doc["provenance"]) == {"python", "executable", "packages", "stack"}


def test_write_result_respects_preexisting_provenance(monkeypatch, tmp_path):
    common = load_common()
    _install_fake_metadata(monkeypatch, common, {})
    out = tmp_path / "r.json"
    common.write_result(str(out), {
        "implementation": "stub", "fit_time_s": 0.1, "peak_rss_gb": 0.1,
        "ll_final": -1.0, "provenance": {"custom": 1},
    })
    assert json.loads(out.read_text())["provenance"] == {"custom": 1}
