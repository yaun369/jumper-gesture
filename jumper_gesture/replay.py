"""离线高清重放一次会话。

    python -m jumper_gesture.replay recordings/session-20261004-160000
    python -m jumper_gesture.replay recordings/session-... --width 1920 --height 1080 --fps 60

读取会话目录里的 events.jsonl（仿真进程记下的每一个按键及其 tick），在同样的场景、
同样的物理频率下从头重跑仿真，逐帧渲染，输出 replay-<宽>x<高>.mp4。
仿真是确定性的，所以动作和现场完全一致，只是画面更清晰、不掉帧。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .sim import replay


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="jumper_gesture.replay", description="离线高清重放一次会话")
    p.add_argument("session", type=Path, help="会话目录（含 events.jsonl 和 session.json）")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--camera-distance", type=float, default=None)
    p.add_argument("--camera-azimuth", type=float, default=None)
    p.add_argument("--camera-elevation", type=float, default=None)
    args = p.parse_args(argv)

    session = args.session.resolve()
    cfg = json.loads((session / "session.json").read_text(encoding="utf-8"))
    cfg.update(width=args.width, height=args.height, fps=args.fps)
    for k in ("distance", "azimuth", "elevation"):
        v = getattr(args, f"camera_{k}")
        if v is not None:
            cfg["camera"][k] = v
    out = session / f"replay-{args.width}x{args.height}.mp4"
    replay(cfg, session / "events.jsonl", out)


if __name__ == "__main__":
    main()
