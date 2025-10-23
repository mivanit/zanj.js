#!/usr/bin/env python3
"""
Generate test ZANJ files for JavaScript tests.
Writes to tests/.temp/ directory.
"""

import zipfile
from pathlib import Path

import numpy as np
from zanj import ZANJ

# Output directory
OUT_DIR = Path(__file__).parent / ".temp"


def generate_basic():
    """Generate basic.zanj with simple inline arrays"""
    data = {
        "small_float": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "small_int": np.array([10, 20, 30], dtype=np.int32),
        "scalar": np.array(42.0, dtype=np.float64),
    }

    z = ZANJ(internal_array_mode="array_b64_meta", external_array_threshold=1000)
    path = z.save(data, OUT_DIR / "basic.zanj")

    # Unzip for direct access
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(OUT_DIR / "basic")

    print(f"✓ Generated {path}")


def generate_all_formats():
    """Generate all-formats.zanj with one of each array format"""
    data = {
        "list_meta": np.array([[1, 2], [3, 4]], dtype=np.int16),
        "b64_meta": np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
        "hex_meta": np.array([0xDE, 0xAD, 0xBE, 0xEF], dtype=np.uint8),
        "zero_dim": np.array(3.14159, dtype=np.float64),
    }

    # Save with different modes to get variety
    z_b64 = ZANJ(internal_array_mode="array_b64_meta", external_array_threshold=1000)
    path = z_b64.save(data, OUT_DIR / "all-formats.zanj")

    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(OUT_DIR / "all-formats")

    print(f"✓ Generated {path}")


def generate_mixed():
    """Generate mixed.zanj with both inline and external arrays"""
    # Small arrays will be inline, large will be external
    data = {
        "inline_small": np.array([1, 2, 3, 4, 5], dtype=np.int32),
        "external_big": np.random.default_rng(42).normal(size=(100, 32)).astype(np.float32),
        "nested": {
            "inline_nested": np.array([0.1, 0.2, 0.3], dtype=np.float64),
            "metadata": {"name": "test", "version": 1},
        },
    }

    # Threshold of 50 elements means inline_small and inline_nested are inline
    z = ZANJ(internal_array_mode="array_b64_meta", external_array_threshold=50)
    path = z.save(data, OUT_DIR / "mixed.zanj")

    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(OUT_DIR / "mixed")

    print(f"✓ Generated {path}")


def generate_edge_cases():
    """Generate edge-cases.zanj with unusual but valid arrays"""
    data = {
        "empty_1d": np.array([], dtype=np.float32),
        "empty_2d": np.array([[], []], dtype=np.int32),
        "single_element": np.array([999], dtype=np.uint32),
        "high_rank": np.ones((2, 2, 2, 2), dtype=np.float32),
        "uint64_max": np.array([2**63 - 1], dtype=np.uint64),
        "negative_int": np.array([-1, -2, -3], dtype=np.int8),
    }

    z = ZANJ(internal_array_mode="array_b64_meta", external_array_threshold=1000)
    path = z.save(data, OUT_DIR / "edge-cases.zanj")

    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(OUT_DIR / "edge-cases")

    print(f"✓ Generated {path}")


def generate_dtypes():
    """Generate dtypes.zanj with all supported dtypes"""
    data = {
        "uint8": np.array([1, 2, 3], dtype=np.uint8),
        "uint16": np.array([100, 200, 300], dtype=np.uint16),
        "uint32": np.array([1000, 2000, 3000], dtype=np.uint32),
        "uint64": np.array([10000, 20000, 30000], dtype=np.uint64),
        "int8": np.array([-1, 0, 1], dtype=np.int8),
        "int16": np.array([-100, 0, 100], dtype=np.int16),
        "int32": np.array([-1000, 0, 1000], dtype=np.int32),
        "int64": np.array([-10000, 0, 10000], dtype=np.int64),
        "float32": np.array([1.5, 2.5, 3.5], dtype=np.float32),
        "float64": np.array([1.123456789, 2.123456789], dtype=np.float64),
    }

    z = ZANJ(internal_array_mode="array_b64_meta", external_array_threshold=1000)
    path = z.save(data, OUT_DIR / "dtypes.zanj")

    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(OUT_DIR / "dtypes")

    print(f"✓ Generated {path}")


def main():
    print("Generating test data...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_basic()
    generate_all_formats()
    generate_mixed()
    generate_edge_cases()
    generate_dtypes()

    print(f"\n✓ All test data generated in {OUT_DIR}")


if __name__ == "__main__":
    main()
