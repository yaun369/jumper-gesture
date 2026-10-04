"""手势识别：YOLO 逐帧检测 + 稳定性过滤，输出"确认的手势事件"。

单帧检测会抖（手在动、光线变），直接拿来控制机器人会乱触发。这里的规则：
- 每帧取置信度最高、且在映射表里的那个手势；
- 同一手势连续 `stable_frames` 帧才算"确认"，确认时触发一次；
- 手势消失 `release_frames` 帧后才"复位"，同一手势才能再次触发
  （一直举着手不会反复触发）；
- 持续动作（如向前走）在确认时开始、复位时结束。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import actions as A


@dataclass
class Detection:
    name: str
    conf: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2（原图坐标）


@dataclass
class Event:
    kind: str  # "trigger" | "hold_on" | "hold_off"
    gesture: str
    action: str


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for g, item in cfg["gestures"].items():
        if item["action"] not in A.ALL_ACTIONS:
            raise SystemExit(f"[config] 手势 {g} 的 action「{item['action']}」不存在，"
                             f"可选：{', '.join(sorted(A.ALL_ACTIONS))}")
    return cfg


class Detector:
    """HaGRIDv2 YOLOv10n 手势检测（Ultralytics 格式）。"""

    def __init__(self, weights: Path, device: str | None = None, imgsz: int = 640) -> None:
        import torch
        from ultralytics import YOLO

        if not Path(weights).is_file():
            raise SystemExit(f"[gesture] 找不到模型 {weights}，先运行 python scripts/download_model.py")
        self.model = YOLO(str(weights))
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda:0" if torch.cuda.is_available() else "cpu")
        self.imgsz = imgsz
        self.names = self.model.names
        # 预热：第一次推理（尤其是 MPS）要编译算子，慢一秒左右，别让它落在第一个手势上
        self.model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=imgsz, device=self.device, verbose=False)

    def __call__(self, bgr: np.ndarray, conf: float) -> list[Detection]:
        r = self.model.predict(bgr, conf=conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        out = []
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                                          r.boxes.conf.tolist()):
            out.append(Detection(self.names[int(c)], float(s), (int(x1), int(y1), int(x2), int(y2))))
        return out


@dataclass
class Stabilizer:
    mapping: dict  # gesture -> {"action", "label"}
    stable_frames: int = 6
    release_frames: int = 8
    _candidate: str | None = None
    _count: int = 0
    _active: str | None = None  # 已确认、尚未复位的手势
    _missing: int = 0
    events: list[Event] = field(default_factory=list)

    def update(self, detections: list[Detection]) -> tuple[Detection | None, list[Event]]:
        """喂入一帧检测结果，返回（本帧采用的检测，本帧产生的事件）。"""
        usable = [d for d in detections if d.name in self.mapping]
        best = max(usable, key=lambda d: d.conf, default=None)
        name = best.name if best else None
        events: list[Event] = []

        if name == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = name, 1

        if self._active is not None:
            if name == self._active:
                self._missing = 0
            else:
                self._missing += 1
                if self._missing >= self.release_frames:
                    action = self.mapping[self._active]["action"]
                    if action in A.HOLDS:
                        events.append(Event("hold_off", self._active, action))
                    self._active = None

        if (name is not None and self._active is None and self._count >= self.stable_frames):
            action = self.mapping[name]["action"]
            events.append(Event("hold_on" if action in A.HOLDS else "trigger", name, action))
            self._active, self._missing = name, 0
        return best, events

    @property
    def progress(self) -> float:
        """当前候选手势的确认进度 0~1（界面上画进度条用）。"""
        if self._candidate is None or self._active == self._candidate:
            return 0.0
        return min(1.0, self._count / self.stable_frames)
