"""The wheel's package list must match the subpackages on disk.

This repository's root *is* the `smart_router` package, so setuptools cannot discover
subpackages and pyproject.toml lists them explicitly. An unlisted subpackage would be
silently missing from the wheel -- importable in development, absent after install. This
test makes that failure loud instead.
"""

from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")

ROOT = Path(__file__).resolve().parents[1]
_SKIP = ("tests", "build", "dist")


def _on_disk() -> set[str]:
    found = {"smart_router"}
    for init in ROOT.rglob("__init__.py"):
        rel = init.parent.relative_to(ROOT)
        if init.parent == ROOT or any(p.startswith(".") or p in _SKIP for p in rel.parts):
            continue
        found.add("smart_router." + ".".join(rel.parts))
    return found


def test_every_subpackage_is_declared_for_the_wheel():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(config["tool"]["setuptools"]["packages"])
    assert declared == _on_disk()


def test_the_package_maps_to_the_repository_root():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["setuptools"]["package-dir"] == {"smart_router": "."}
