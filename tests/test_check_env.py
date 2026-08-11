"""check_env.py: install specs + installed==intended verification against pins.toml.

Hermetic — installed distributions are faked, no venv/network. Pure stdlib
(tomllib on 3.11+), so it does not touch the AMICA compute path.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def load_check_env():
    path = REPO / "benchmark" / "cc_benchmark" / "check_env.py"
    spec = importlib.util.spec_from_file_location("check_env", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PINS = REPO / "benchmark" / "cc_benchmark" / "pins.toml"


class _FakeDist:
    def __init__(self, name, version, url=None, commit=None):
        self.version = version
        self.metadata = {"Name": name}
        self._url = url
        self._commit = commit

    def read_text(self, filename):
        if filename == "direct_url.json" and self._url:
            info = {"url": self._url}
            if self._commit:
                info["vcs_info"] = {"vcs": "git", "commit_id": self._commit}
            return json.dumps(info)
        return None


def test_canon_url_normalizes():
    ce = load_check_env()
    a = ce.canon_url("git+https://github.com/Owner/Repo.git@abc123")
    b = ce.canon_url("https://github.com/owner/repo/")
    assert a == b == "https://github.com/owner/repo"


def test_canon_url_preserves_ssh_user_at_host():
    ce = load_check_env()
    # the user@host '@' must survive; only a trailing @rev is stripped
    assert ce.canon_url("ssh://git@github.com/owner/repo.git@deadbeef") == \
        "ssh://git@github.com/owner/repo"
    assert ce.canon_url("git@github.com:owner/repo.git") == "git@github.com:owner/repo"


def test_specs_version_pins_and_git_pins(capsys):
    ce = load_check_env()
    # competitors are PyPI version pins -> `name==version`
    assert ce.cmd_specs(ce.load_venv(PINS, "competitors")) == 0
    assert capsys.readouterr().out.strip().splitlines() == [
        "pyamica==0.3.0", "amica-python==0.1.1",
    ]
    # neuromechanist is a git commit pin -> `git+url@commit`
    assert ce.cmd_specs(ce.load_venv(PINS, "neuromechanist")) == 0
    assert capsys.readouterr().out.strip() == \
        "git+https://github.com/sccn/pAMICA.git@526aa3231623490ea21ef9c45acbb50730929622"


def test_unknown_venv_raises():
    ce = load_check_env()
    with pytest.raises(SystemExit):
        ce.load_venv(PINS, "does-not-exist")


def _patch_installed(monkeypatch, ce, dists):
    monkeypatch.setattr(ce.im, "distributions", lambda: iter(dists))


def _patch_versions(monkeypatch, ce, versions):
    def fake_version(name):
        if name in versions:
            return versions[name]
        raise ce.im.PackageNotFoundError(name)
    monkeypatch.setattr(ce.im, "version", fake_version)
    monkeypatch.setattr(ce.im, "distributions", lambda: iter([]))  # version path ignores these


def test_verify_version_pin_ok_including_local_suffix(monkeypatch):
    ce = load_check_env()
    venv = ce.load_venv(PINS, "competitors")
    # the +computecanada local segment must still match the pinned base version
    _patch_versions(monkeypatch, ce, {"pyamica": "0.3.0", "amica-python": "0.1.1+computecanada"})
    assert ce.cmd_verify(venv) == 0


def test_verify_version_pin_mismatch(monkeypatch, capsys):
    ce = load_check_env()
    venv = ce.load_venv(PINS, "competitors")
    _patch_versions(monkeypatch, ce, {"pyamica": "0.4.0", "amica-python": "0.1.1"})
    assert ce.cmd_verify(venv) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_verify_version_pin_missing(monkeypatch, capsys):
    ce = load_check_env()
    venv = ce.load_venv(PINS, "competitors")
    _patch_versions(monkeypatch, ce, {"pyamica": "0.3.0"})  # amica-python absent
    assert ce.cmd_verify(venv) == 1
    assert "MISSING" in capsys.readouterr().out


def test_verify_commit_pin_ok(monkeypatch):
    ce = load_check_env()
    venv = ce.load_venv(PINS, "neuromechanist")
    dists = [_FakeDist("pyAMICA", "0.1.dev0", "git+https://github.com/sccn/pAMICA.git",
                       "526aa3231623490ea21ef9c45acbb50730929622")]
    _patch_installed(monkeypatch, ce, dists)
    assert ce.cmd_verify(venv) == 0


def test_verify_commit_pin_fails_on_missing(monkeypatch, capsys):
    ce = load_check_env()
    venv = ce.load_venv(PINS, "neuromechanist")
    _patch_installed(monkeypatch, ce, [])  # nothing installed / clobbered
    assert ce.cmd_verify(venv) == 1
    assert "MISSING" in capsys.readouterr().out


def test_every_pins_venv_pkg_has_a_pin():
    ce = load_check_env()
    doc = ce._load_toml(PINS)
    for venv in doc["venv"]:
        assert venv["packages"], venv["name"]
        for pkg in venv["packages"]:
            # each package is pinned by a PyPI version OR a full 40-char git SHA
            assert pkg.get("version") or len(pkg.get("commit", "")) == 40, \
                (venv["name"], pkg["name"])
