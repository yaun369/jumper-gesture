"""仿真进程：运行官方 jumper.app，接收动作指令，渲染画面。

做法是复用官方 `scripts/play.py --app` 的全部搭建流程（环境、部署控制器、ONNX 策略），
只把它的主循环换成我们自己的：
- 按真实时间推进仿真；
- 每个控制周期把动作指令翻译成按键（`driver.Driver`），用官方的
  `mjrl.viewer.keys.dispatch()` 交给控制器——和在 MuJoCo 窗口里按键盘完全一样；
- 一个单独的渲染线程按固定帧率把画面写进共享内存，交给主进程显示。

另外跳过了世界里的 dToF 深度传感器：`play --app` 下控制器只读关节和 IMU，
没有任何东西读它，而它每步要打 2268 根射线，占仿真耗时的六成以上。

`replay()` 用会话日志离线重放，以任意分辨率渲染，用于成片。
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np


def _prepare_jumper(jumper_root: Path) -> None:
    """让官方仓库的脚本可以被 import，并跳过用不到的世界传感器。"""
    jumper_root = jumper_root.resolve()
    if not (jumper_root / "scripts" / "play.py").is_file():
        raise SystemExit(f"[sim] 找不到官方 jumper 仓库：{jumper_root}（用 --jumper 或 JUMPER_ROOT 指定）")
    os.chdir(jumper_root)
    sys.path.insert(0, str(jumper_root / "scripts"))
    from mjrl.backend import native_sim

    native_sim.NativeSimulation.sense = lambda self: None


def _play_argv(cfg: dict) -> list[str]:
    argv = ["play.py", "--app", str(cfg["app"]), "--headless",
            "--backend", "native", "--device", "cpu",
            "--physics-hz", str(cfg["physics_hz"])]
    if cfg.get("scene"):
        argv += ["--scene", cfg["scene"]]
    return argv


class Renderer:
    """离屏渲染跳跳，相机跟随机身位置和朝向。

    azimuth 是相对机身朝向的水平角：180 = 正面（机身 +x 是前方），0 = 背后，
    90 / 270 = 两侧。机身转向时相机平滑地跟着转，跳跳始终保持同一个角度面对镜头。
    """

    YAW_FOLLOW = 0.15  # 每帧朝当前机身朝向靠近的比例，越小相机越稳、跟得越慢

    def __init__(self, model, width: int, height: int, camera: dict) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        free = [model.jnt_bodyid[j] for j in range(model.njnt)
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        cam.trackbodyid = int(free[0]) if free else 0
        cam.distance = camera.get("distance", 1.1)
        cam.elevation = camera.get("elevation", -18.0)
        self.cam = cam
        self.azimuth = camera.get("azimuth", 180.0)
        self._yaw: float | None = None

    def _body_yaw(self) -> float:
        """机身前方（body +x）在水平面上的朝向，单位度。"""
        R = self.data.xmat[self.cam.trackbodyid].reshape(3, 3)
        return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    def render(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self._mujoco.mj_forward(self.model, self.data)
        yaw = self._body_yaw()
        if self._yaw is None:
            self._yaw = yaw
        else:  # 平滑跟随，按最短方向转，避免 ±180° 处跳变
            self._yaw += self.YAW_FOLLOW * ((yaw - self._yaw + 180.0) % 360.0 - 180.0)
        self.cam.azimuth = self._yaw + self.azimuth
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()


def _key_codes() -> dict[str, int]:
    from controller import vocabulary

    return {name: c["glfw"] for name, c in vocabulary.key_codes().items()}


def run(cfg: dict, commands, status, shm_name: str, frame_seq, stop) -> None:
    """仿真进程入口（由主进程用 multiprocessing 启动）。

    commands: 主进程发来的指令 {"op": "trigger"|"hold", "action": ..., "on": bool}
    status:   发回主进程的状态 {"mode", "running", "rtf", "t", "note", "ready"}
    shm_name: 共享内存名字，存放一帧 RGB 画面（height × width × 3）
    """
    from multiprocessing import shared_memory

    _prepare_jumper(Path(cfg["jumper"]))
    import play
    import torch

    from jumper_gesture.driver import Driver
    from mjrl.viewer import keys

    codes = _key_codes()
    shm = shared_memory.SharedMemory(name=shm_name)
    frame = np.ndarray((cfg["height"], cfg["width"], 3), dtype=np.uint8, buffer=shm.buf)
    log = open(cfg["log"], "w", encoding="utf-8") if cfg.get("log") else None

    def loop(env, policy, viewer, max_steps, speed, readout, monitor):  # noqa: ARG001
        unwrapped = env.unwrapped
        sim = unwrapped.sim
        dt = unwrapped.step_dt
        model, live = sim.env_mjdata(0)
        lock = threading.Lock()
        state = {"qpos": live.qpos.copy(), "qvel": live.qvel.copy()}

        def render_loop() -> None:
            r = Renderer(sim.render_model, cfg["width"], cfg["height"], cfg.get("camera", {}))
            period = 1.0 / cfg["fps"]
            nxt = time.perf_counter()
            while not stop.is_set():
                with lock:
                    qpos, qvel = state["qpos"].copy(), state["qvel"].copy()
                frame[:] = r.render(qpos, qvel)
                with frame_seq.get_lock():
                    frame_seq.value += 1
                nxt += period
                time.sleep(max(0.0, nxt - time.perf_counter()))

        threading.Thread(target=render_loop, daemon=True).start()

        driver = Driver()
        obs = env.get_observations()
        tick = 0
        t0 = time.perf_counter()
        last_status = 0.0
        last_mode = None
        status.put({"ready": True})
        while not stop.is_set():
            mode = policy.fsm.mode
            while True:
                try:
                    cmd = commands.get_nowait()
                except queue.Empty:
                    break
                if cmd["op"] == "trigger":
                    driver.request(cmd["action"], mode, tick)
                elif cmd["op"] == "hold":
                    driver.hold(cmd["action"], cmd["on"], mode, tick)
                if log:
                    log.write(json.dumps({"tick": tick, "cmd": cmd}, ensure_ascii=False) + "\n")
            for name, down in driver.due(mode, tick):
                keys.dispatch(codes[name], down)
                if log:
                    log.write(json.dumps({"tick": tick, "key": name, "down": down}) + "\n")
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
            with lock:
                state["qpos"][:] = live.qpos
                state["qvel"][:] = live.qvel
            tick += 1
            # 按真实时间推进：落后了就不等，领先了就睡到下一个周期
            ahead = t0 + tick * dt - time.perf_counter()
            if ahead > 0:
                time.sleep(ahead)
            now = time.perf_counter()
            mode = policy.fsm.mode
            if mode != last_mode or now - last_status > 0.25:
                status.put({"mode": mode, "running": bool(policy.fsm.running_policy),
                            "rtf": tick * dt / max(1e-6, now - t0), "t": tick * dt,
                            "note": driver.note})
                last_mode, last_status = mode, now
        if log:
            log.close()

    play._loop = loop
    sys.argv = _play_argv(cfg)
    try:
        play.main()
    finally:
        shm.close()


def replay(cfg: dict, events: Path, out: Path) -> None:
    """离线重放一次会话：按日志里记录的 tick 原样发送按键，逐帧高清渲染。

    不需要实时，所以分辨率可以开到 1080p / 4K；输出只有仿真画面，方便剪辑时和实拍对齐。
    """
    _prepare_jumper(Path(cfg["jumper"]))
    import mediapy
    import play
    import torch

    from mjrl.viewer import keys

    codes = _key_codes()
    edges: dict[int, list[tuple[str, bool]]] = {}
    last_tick = 0
    for line in events.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        last_tick = max(last_tick, rec["tick"])
        if "key" in rec:
            edges.setdefault(rec["tick"], []).append((rec["key"], rec["down"]))

    def loop(env, policy, viewer, max_steps, speed, readout, monitor):  # noqa: ARG001
        unwrapped = env.unwrapped
        sim = unwrapped.sim
        hz = 1.0 / unwrapped.step_dt
        every = hz / cfg["fps"]
        total = last_tick + int(cfg.get("tail_s", 3.0) * hz)
        _, live = sim.env_mjdata(0)
        r = Renderer(sim.render_model, cfg["width"], cfg["height"], cfg.get("camera", {}))
        obs = env.get_observations()
        next_frame = 0.0
        with mediapy.VideoWriter(str(out), shape=(cfg["height"], cfg["width"]), fps=cfg["fps"]) as w:
            for tick in range(total):
                for name, down in edges.get(tick, []):
                    keys.dispatch(codes[name], down)
                with torch.inference_mode():
                    obs, _, _, _ = env.step(policy(obs))
                if tick >= next_frame:
                    w.add_image(r.render(live.qpos.copy(), live.qvel.copy()))
                    next_frame += every
                if tick % int(hz) == 0:
                    print(f"\r[replay] {tick / hz:6.1f}s / {total / hz:.1f}s", end="", flush=True)
        print(f"\n[replay] 已写出 {out}")

    play._loop = loop
    sys.argv = _play_argv(cfg)
    play.main()
