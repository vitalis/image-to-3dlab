"""Hero and back-view stop when the remover file is missing."""

import pytest

from scripts.hero_route import ensure_matte as hero_matte
from scripts.make_back_view import ensure_matte as back_matte


def test_hero_names_the_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("U2NET_HOME", str(tmp_path))
    with pytest.raises(SystemExit, match=r"u2net \(176 MB\)"):
        hero_matte()


def test_back_view_names_the_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("U2NET_HOME", str(tmp_path))
    with pytest.raises(SystemExit, match=r"bootstrap_matte.py"):
        back_matte()
