from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _ensure_dir(p: Path) -> None:
    """ensure directory exists and is a dir"""
    p.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, text: str) -> None:
    """write text to file atomically-ish"""
    tmp: Path = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _np_save(path: Path, arr: np.ndarray) -> None:
    """save array to .npy (no pickle)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr, allow_pickle=False)


def build_zanj_unzipped(
    out_dir: Path,
    *,
    n: int = 200000,
    d: int = 32,
    seed: int = 0,
) -> None:
    """build an unzipped zanj folder with refs to json and npy

    extended summary

    # Parameters:
     - `out_dir : Path`
        directory to create; will contain __zanj__.json, info.json, big_array.npy
     - `n : int`
        number of rows in the demo array (defaults to 200000)
     - `d : int`
        number of columns in the demo array (defaults to 32)
     - `seed : int`
        RNG seed for reproducibility (defaults to 0)

    # Returns:
     - `None`
        writes files into out_dir

    # Modifies:
     - `out_dir` : creates files in place

    # Usage:

    ```python
    >>> build_zanj_unzipped(Path("site/data/zanj_demo"))
    ```

    # Raises:
     - `OSError` : filesystem errors
    """
    _ensure_dir(out_dir)

    # Small JSON sidecar
    info: dict[str, Any] = {
        "title": "zanj demo",
        "description": "uncompressed zanj folder for frontend lazy loading",
        "schema": {
            "big_array": {"dtype": "float32", "shape": [n, d], "path": "big_array.npy"},
        },
    }
    _write_text(out_dir / "info.json", json.dumps(info, indent=2))

    # Big array
    rng: np.random.Generator = np.random.default_rng(seed)
    big: np.ndarray = rng.normal(loc=0.0, scale=1.0, size=(n, d)).astype(np.float32)
    _np_save(out_dir / "big_array.npy", big)

    # Root zanj JSON with lazy refs
    root: dict[str, Any] = {
        "version": 1,
        "info": {"$ref": "info.json", "format": "json"},
        "big_array": {"$ref": "big_array.npy", "format": "npy"},
    }
    _write_text(out_dir / "__zanj__.json", json.dumps(root, indent=2))


if __name__ == "__main__":
    # Example: write to ./site/data/zanj_demo
    target: Path = Path("site") / "data" / "zanj_demo"
    build_zanj_unzipped(target)
    print(f"Wrote demo to {target}")
