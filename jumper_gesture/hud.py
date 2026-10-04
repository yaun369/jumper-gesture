"""画面合成：摄像头（带识别框）+ 仿真画面 + 底部信息条，中文用 PIL 绘制。"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import actions as A

_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",           # macOS
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",                    # Windows
]

ACCENT = (255, 196, 0)      # 识别框、高亮
RED = (230, 57, 70)         # 跳跳红
WHITE = (255, 255, 255)
GREY = (170, 170, 170)


def _font(size: int) -> ImageFont.ImageFont:
    for p in _FONT_CANDIDATES:
        if Path(p).is_file():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """等比缩放后居中裁切到 w×h。"""
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    nw, nh = int(round(iw * s)), int(round(ih * s))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x, y = (nw - w) // 2, (nh - h) // 2
    return img[y:y + h, x:x + w]


class Hud:
    def __init__(self, layout: str = "horizontal") -> None:
        self.layout = layout
        if layout == "vertical":       # 竖屏 9:16，适合抖音
            self.size = (720, 1280)
            self.panel = (720, 560)
            self.bar_h = 160
        else:                           # 横屏 16:9
            self.size = (1280, 720)
            self.panel = (640, 600)
            self.bar_h = 120
        self.f_big = _font(40)
        self.f_mid = _font(28)
        self.f_small = _font(20)
        self.flash_until = 0.0
        self.flash_text = ""

    def flash(self, text: str, seconds: float = 2.0) -> None:
        self.flash_text, self.flash_until = text, time.time() + seconds

    def compose(self, cam_bgr: np.ndarray, det, mapping: dict, progress: float,
                sim_rgb: np.ndarray, status: dict, recording: bool) -> np.ndarray:
        W, H = self.size
        pw, ph = self.panel
        # 摄像头：先画框再镜像，用户看到的是镜子里的自己
        cam = cam_bgr.copy()
        if det is not None:
            x1, y1, x2, y2 = det.box
            cv2.rectangle(cam, (x1, y1), (x2, y2), ACCENT[::-1], 4)
        cam = cv2.flip(cam, 1)
        cam = fit(cam, pw, ph)
        sim = fit(cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR), pw, ph)

        canvas = np.full((H, W, 3), 18, np.uint8)
        if self.layout == "vertical":
            canvas[0:ph, :] = cam
            canvas[ph:2 * ph, :] = sim
            cam_org, sim_org = (0, 0), (0, ph)
        else:
            canvas[0:ph, 0:pw] = cam
            canvas[0:ph, pw:2 * pw] = sim
            cam_org, sim_org = (0, 0), (pw, 0)

        img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)

        # 面板标题
        d.text((cam_org[0] + 16, cam_org[1] + 12), "我（摄像头 + YOLO 手势识别）", font=self.f_small, fill=WHITE)
        d.text((sim_org[0] + 16, sim_org[1] + 12), "跳跳（官方开源仿真）", font=self.f_small, fill=WHITE)
        if recording:
            d.ellipse((W - 40, 14, W - 20, 34), fill=RED)

        # 识别结果标签 + 确认进度条
        if det is not None:
            item = mapping.get(det.name, {})
            tag = f"{item.get('label', det.name)}  {det.conf:.2f}"
            # 标签固定画在摄像头面板左下角，避免跟着框乱跳
            y = cam_org[1] + ph - 70
            d.rectangle((cam_org[0] + 12, y, cam_org[0] + 36 + int(d.textlength(tag, font=self.f_mid)), y + 52),
                        fill=(0, 0, 0))
            d.text((cam_org[0] + 24, y + 8), tag, font=self.f_mid, fill=ACCENT)
            if progress > 0:
                d.rectangle((cam_org[0] + 12, y + 54, cam_org[0] + 12 + int(260 * progress), y + 60),
                            fill=ACCENT)

        # 底部信息条
        bar_y = H - self.bar_h
        d.rectangle((0, bar_y, W, H), fill=(28, 28, 32))
        mode = status.get("mode", "启动中")
        state = "" if status.get("running", True) else "（过渡中）"
        d.text((24, bar_y + 18), "跳跳正在：", font=self.f_mid, fill=GREY)
        d.text((24 + 150, bar_y + 10), A.label(mode) + state, font=self.f_big, fill=RED)
        if time.time() < self.flash_until:
            d.text((24, bar_y + 70), self.flash_text, font=self.f_mid, fill=ACCENT)
        rtf = status.get("rtf")
        if rtf:
            d.text((W - 230, H - 34), f"仿真速度 {rtf:.2f}x 实时", font=self.f_small, fill=GREY)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
