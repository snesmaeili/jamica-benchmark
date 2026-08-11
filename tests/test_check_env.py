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


def test_specs_prints_pinned_git_urls(capsys):
    ce = load_check_env()
    venv = ce.load_venv(PINS, "competitors")
    assert ce.cmd_specs(venv) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out == [
        "git+https://github.com/DerAndereJohannes/pyamica.git@a8a4d7e0ad14a88cf2cabeff5094cd0c8a262536",
        "git+https://github.com/scott-huberty/amica-python.git@e15e15888a5f6d366c6b16b1884bd373c319c085",
    ]


def test_unknown_venv_raises():
    ce = load_check_env()
    with pytest.raises(SystemExit):
        ce.load_venv(PINS, "does-not-exist")


def _patch_installed(monkeypatch, ce, dists):
    monkeypatch.setattr(ce.im, "distributions", lambda: iter(dists))


def test_verify_ok_when_commit_matches(monkeypatch):
    ce = load_check_env()
    venv = ce.load_venv(PINS, "competitors")
    dists = [
        _FakeDist("pyamica", "0.4.1", "https://github.com/DerAndereJohannes/pyamica.git",
                  "a8a4d7e0ad14a88cf2cabeff5094cd0c8a262536"),
        # scott's install reports a canonicalized name; matching is by URL, not name.
        _FakeDist("amica-python", "0.2.0", "git+https://github.com/scott-huberty/amica-python.git",
                  "e15e15888a5f6d366c6b16b1884bd373c319c085"),
    ]
    _patch_installed(monkeypatch, ce, dists)
    assert ce.cmd_verify(venv) == 0


def test_verify_fails_on_commit_mismatch(monkeypatch, capsys):
    ce = load_check_env()
    venv = ce.load_venv(PINS, "competitors")
    dists = [
        _FakeDist("pyamica", "0.4.1", "https://github.com/DerAndereJohannes/pyamica.git",
                  "deadbeef" * 5),  # wrong commit
        _FakeDist("amica-python", "0.2.0", "https://github.com/scott-huberty/amica-python.git",
                  "e15e15888a5f6d366c6b16b1884bd373c319c085"),
    ]
    _patch_installed(monkeypatch, ce, dists)
    assert ce.cmd_verify(venv) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_verify_fails_on_missing(monkeypatch, capsys):
    ce = load_check_env()
    venv = ce.load_venv(PINS, "neuromechanist")
    _patch_installed(monkeypatch, ce, [])  # nothing installed / clobbered
    assert ce.cmd_verify(venv) == 1
    assert "MISSING" in capsys.readouterr().out


def test_every_pins_venv_has_packages():
    ce = load_check_env()
    doc = ce._load_toml(PINS)
    for venv in doc["venv"]:
        assert venv["packages"], venv["name"]
        for pkg in venv["packages"]:
            assert len(pkg["commit"]) == 40, (venv["name"], pkg["name"])  # full SHA
