"""手势控制跳跳：主程序。

主进程负责摄像头、YOLO 手势识别、画面合成与录制；仿真在子进程里跑（见 sim.py），
两边通过队列传指令/状态、通过共享内存传画面。分成两个进程是因为 macOS 上
OpenCV 窗口必须在主线程，而仿真和识别都吃 CPU，分开后互不拖累。

    python -m jumper_gesture                  # 默认摄像头
    python -m jumper_gesture --layout vertical --record
    python -m jumper_gesture --source demo.mp4  # 用视频文件代替摄像头
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import time
from datetime import datetime
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np

from . import actions as A
from .gestures import Detector, Stabilizer, load_config
from .hud import Hud

REPO = Path(__file__).resolve().parent.parent


def default_jumper() -> Path:
    env = os.environ.get("JUMPER_ROOT")
    return Path(env) if env else REPO.parent / "jumper"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="jumper_gesture", description="用手势控制螃蟹机器人跳跳（仿真）")
    p.add_argument("--source", default="0", help="摄像头编号（默认 0）或视频文件路径")
    p.add_argument("--config", type=Path, default=REPO / "config" / "gestures.yaml", help="手势映射表")
    p.add_argument("--model", type=Path, default=REPO / "models" / "YOLOv10n_gestures.pt", help="手势模型权重")
    p.add_argument("--device", default=None, help="YOLO 推理设备：mps / cuda:0 / cpu，默认自动")
    p.add_argument("--jumper", type=Path, default=default_jumper(),
                   help="官方 jumper 仓库路径（默认 ../jumper，或环境变量 JUMPER_ROOT）")
    p.add_argument("--app", type=Path, default=None, help="动作包，默认用官方示例 jumper.app")
    p.add_argument("--scene", default="daylight",
                   help="仿真场景：daylight（默认）/ studio / beach / default 等，完整列表见 README")
    p.add_argument("--physics-hz", type=int, default=200, help="物理频率，200 与训练一致，1000 更精细但更慢")
    p.add_argument("--layout", choices=["horizontal", "vertical"], default="horizontal",
                   help="horizontal 横屏 16:9；vertical 竖屏 9:16（抖音）")
    p.add_argument("--fps", type=int, default=30, help="仿真渲染与录制帧率")
    p.add_argument("--record", action="store_true", help="启动后立刻开始录制（窗口里按 R 也可以开关）")
    p.add_argument("--out", type=Path, default=REPO / "recordings", help="录制输出目录")
    p.add_argument("--camera-distance", type=float, default=1.1)
    p.add_argument("--camera-azimuth", type=float, default=150.0)
    p.add_argument("--camera-elevation", type=float, default=-18.0)
    p.add_argument("--no-window", action="store_true", help="不开窗口（配合 --duration 做自动测试）")
    p.add_argument("--duration", type=float, default=None, help="运行这么多秒后自动退出")
    return p.parse_args(argv)


class Recorder:
    """按墙钟时间写帧：画面卡了就重复上一帧，保证视频时长和真实时长一致。"""

    def __init__(self, path: Path, size: tuple[int, int], fps: int) -> None:
        self.w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.fps, self.t0, self.n, self.path = fps, time.perf_counter(), 0, path

    def write(self, frame: np.ndarray) -> None:
        due = int((time.perf_counter() - self.t0) * self.fps) + 1
        while self.n < due:
            self.w.write(frame)
            self.n += 1

    def close(self) -> None:
        self.w.release()


def open_source(src: str):
    cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)
    if src.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit(f"[app] 打不开视频源 {src}。macOS 需要在「系统设置 → 隐私与安全性 → 摄像头」"
                         "里允许你用的终端 / VS Code 访问摄像头")
    return cap


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    mapping = cfg["gestures"]
    rec_cfg = cfg.get("recognition", {})
    app_path = (args.app or args.jumper / "out" / "bundle_example" / "jumper.app").resolve()

    session = args.out / datetime.now().strftime("session-%Y%m%d-%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    hud = Hud(args.layout)
    pw, ph = hud.panel

    # ── 仿真子进程 ──
    ctx = mp.get_context("spawn")
    shm = shared_memory.SharedMemory(create=True, size=pw * ph * 3)
    sim_frame = np.ndarray((ph, pw, 3), dtype=np.uint8, buffer=shm.buf)
    sim_frame[:] = 30
    seq = ctx.Value("l", 0)
    stop = ctx.Event()
    commands, status_q = ctx.Queue(), ctx.Queue()
    sim_cfg = {
        "jumper": str(args.jumper.resolve()), "app": str(app_path), "scene": args.scene,
        "physics_hz": args.physics_hz, "width": pw, "height": ph, "fps": args.fps,
        "log": str(session / "events.jsonl"),
        "camera": {"distance": args.camera_distance, "azimuth": args.camera_azimuth,
                   "elevation": args.camera_elevation},
    }
    (session / "session.json").write_text(
        json.dumps({k: v for k, v in sim_cfg.items() if k != "log"},
                                 ensure_ascii=False, indent=1), encoding="utf-8")
    from . import sim

    proc = ctx.Process(target=sim.run, args=(sim_cfg, commands, status_q, shm.name, seq, stop),
                       daemon=True)
    proc.start()

    # ── 手势识别 ──
    detector = Detector(args.model, args.device, rec_cfg.get("imgsz", 640))
    stab = Stabilizer(mapping, rec_cfg.get("stable_frames", 6), rec_cfg.get("release_frames", 8))
    conf = rec_cfg.get("confidence", 0.5)
    print(f"[app] YOLO 设备：{detector.device}；会话目录：{session}")

    cap = open_source(args.source)
    is_file = not args.source.isdigit()
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    status: dict = {"mode": "启动中"}
    recorder = cam_recorder = None
    manual = list(mapping.items())  # 键盘 1~9 手动模拟手势，方便没有摄像头时调试
    window = "jumper-gesture"
    if not args.no_window:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, *hud.size)

    def toggle_record() -> None:
        nonlocal recorder, cam_recorder
        if recorder is None:
            stamp = datetime.now().strftime("%H%M%S")
            recorder = Recorder(session / f"composite-{stamp}.mp4", hud.size, args.fps)
            cam_recorder = None  # 原始摄像头画面在拿到第一帧尺寸时再建
            commands.put({"op": "mark", "name": f"record_start {stamp}"})
            hud.flash("开始录制")
            print(f"[app] 开始录制 → {recorder.path}")
        else:
            recorder.close()
            if cam_recorder:
                cam_recorder.close()
            commands.put({"op": "mark", "name": "record_stop"})
            print(f"[app] 录制结束：{recorder.path}")
            recorder = cam_recorder = None
            hud.flash("录制结束")

    def send(event_kind: str, action: str, gesture_label: str) -> None:
        if event_kind == "trigger":
            commands.put({"op": "trigger", "action": action})
            hud.flash(f"识别到「{gesture_label}」→ 跳跳：{A.label(action)}")
        else:
            commands.put({"op": "hold", "action": action, "on": event_kind == "hold_on"})
            if event_kind == "hold_on":
                hud.flash(f"识别到「{gesture_label}」→ 跳跳：{A.label(action)}（保持手势）")

    # 等仿真加载完（首次约 10~20 秒），期间不读摄像头，免得手势指令堆在队列里
    t_wait = time.perf_counter()
    while True:
        try:
            if status_q.get(timeout=0.1).get("ready"):
                break
        except queue.Empty:
            pass
        if not proc.is_alive():
            raise SystemExit("[app] 仿真进程启动失败，往上翻看报错")
        if time.perf_counter() - t_wait > 180:
            raise SystemExit("[app] 仿真 3 分钟都没加载完，检查 --jumper 路径和环境")
        if not args.no_window:
            loading = np.full((hud.size[1], hud.size[0], 3), 18, np.uint8)
            cv2.putText(loading, f"loading simulation... {time.perf_counter() - t_wait:.0f}s",
                        (40, hud.size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
            cv2.imshow(window, loading)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                stop.set()
                return
    print(f"[app] 仿真就绪（{time.perf_counter() - t_wait:.1f}s），开始识别")
    if is_file:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if args.record:
        toggle_record()
    t_start = next_t = time.perf_counter()
    try:
        while True:
            ok, cam = cap.read()
            if not ok:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("[app] 摄像头读取失败")
                break
            dets = detector(cam, conf)
            best, events = stab.update(dets)
            for ev in events:
                send(ev.kind, ev.action, mapping[ev.gesture]["label"])

            while True:
                try:
                    msg = status_q.get_nowait()
                except queue.Empty:
                    break
                status.update(msg)
            if not proc.is_alive():
                raise SystemExit("[app] 仿真进程退出了，往上翻看报错")

            frame = hud.compose(cam, best, mapping, stab.progress, sim_frame.copy()[..., :3],
                                status, recorder is not None)
            if recorder is not None:
                recorder.write(frame)
                if cam_recorder is None:
                    h, w = cam.shape[:2]
                    cam_recorder = Recorder(recorder.path.with_name(
                        recorder.path.name.replace("composite", "camera")), (w, h), args.fps)
                    cam_recorder.t0 = recorder.t0
                cam_recorder.write(cam)

            if not args.no_window:
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    toggle_record()
                if ord("1") <= key <= ord("9") and key - ord("1") < len(manual):
                    g, item = manual[key - ord("1")]
                    kind = "hold_on" if item["action"] in A.HOLDS else "trigger"
                    send(kind, item["action"], item["label"] + "（键盘）")
            if is_file:  # 视频文件按原速播放，不然会被读得飞快
                next_t += 1.0 / file_fps
                time.sleep(max(0.0, next_t - time.perf_counter()))
            if args.duration and time.perf_counter() - t_start > args.duration:
                break
    finally:
        if recorder is not None:
            toggle_record()
        stop.set()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
        cap.release()
        cv2.destroyAllWindows()
        shm.close()
        shm.unlink()
        print(f"[app] 本次会话：{session}（events.jsonl 可用于高清重放：python -m jumper_gesture.replay {session}）")
