import numpy as np
import zipfile
from pathlib import Path
from zanj import ZANJ


def make_demo(out_dir: Path) -> None:
    # 1. make some data
    arr = np.random.default_rng(0).normal(size=(2000, 32, 32)).astype(np.float32)
    data = {
        "meta": {"title": "zanj demo"},
        "big_list": [f"item {i}" for i in range(1000)],
        "big_array": arr,
        # Test inline array formats
        "inline_arrays": {
            "small_list_meta": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "small_b64_meta": np.array([10, 20, 30, 40], dtype=np.int32),
            "small_hex_meta": np.array([1, 2, 3], dtype=np.uint8),
            "matrix_2x3": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16),
            "zero_dim": np.array(42.0, dtype=np.float32),
        }
    }

    # 2. save to a .zanj
    # Use a low threshold to force big_array external, but keep inline ones inline
    # Use array_b64_meta for inline arrays to test base64 deserialization
    z = ZANJ(internal_array_mode="array_b64_meta", external_array_threshold=100)
    path = "demo.zanj"
    z.save(data, path)

    # 3. unzip into the same directory for frontend to read directly
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(out_dir)


if __name__ == "__main__":
    make_demo(Path("demo"))
