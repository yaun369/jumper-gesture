"""下载 HaGRIDv2 的 YOLOv10n 手势检测模型（约 22 MB）到 models/。

模型来自 HaGRID 项目（https://github.com/hukenovs/hagrid），
使用其许可证（署名 + 保留条件，基于 CC BY-SA 4.0 改写），使用时请保留出处。
"""

import sys
import urllib.request
from pathlib import Path

URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/YOLOv10n_gestures.pt"
OUT = Path(__file__).resolve().parent.parent / "models" / "YOLOv10n_gestures.pt"


def main() -> None:
    if OUT.is_file() and OUT.stat().st_size > 10_000_000:
        print(f"已存在：{OUT}")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载 {URL}\n  → {OUT}")

    def hook(n, block, total):
        if total > 0:
            sys.stdout.write(f"\r  {min(100, n * block * 100 // total):3d}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(URL, OUT, hook)
    print("\n完成")


if __name__ == "__main__":
    main()
