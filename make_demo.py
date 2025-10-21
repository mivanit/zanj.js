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
    }

    # 2. save to a .zanj (force even small things external to demo behavior)
    z = ZANJ(external_array_threshold=10)
    path = "demo.zanj"
    z.save(data, path)

    # 3. unzip into the same directory for frontend to read directly
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(out_dir)


if __name__ == "__main__":
    make_demo(Path("demo"))
