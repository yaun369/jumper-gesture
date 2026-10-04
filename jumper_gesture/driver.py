"""把"让跳跳做某个动作"翻译成按键边沿，按控制周期（tick）排期。

在仿真进程里运行，每个 tick 读一次控制器的真实模式，所以能处理这些情况：
- 固定动作只能从站立进入：正在跳舞时比了"挥手"，先单按 Ctrl 退出舞蹈，回到站立后再按 1；
- 跳跃不能打断：跳跃期间的手势直接忽略；
- 已经在做的动作不重复触发（官方开关是 toggle，再按一次反而会退出）。

纯 Python，不依赖仿真，便于单独测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import actions as A

# 同一组按键里相邻两个边沿间隔的 tick 数。控制器每个 tick 读一次按键，
# 间隔 2 个 tick 能保证"按下"和"松开"被看成两次独立的事件。
EDGE_GAP = 2
# 退出舞蹈回到站立后，再等这么多 tick 才发下一个动作（让模式稳定下来）。
SETTLE_TICKS = 10


@dataclass
class Driver:
    #: 待发送的边沿：(tick, 按键名, 是否按下)
    pending: list[tuple[int, str, bool]] = field(default_factory=list)
    #: 等站立后再执行的动作
    deferred: str | None = None
    deferred_since: int | None = None
    #: 正在按住的持续动作
    holding: str | None = None
    #: 最近一次说明（给界面显示）
    note: str = ""

    def _schedule(self, tick: int, edges: list[tuple[str, bool]]) -> None:
        start = max([tick] + [t + EDGE_GAP for t, _, _ in self.pending])
        for k, (name, down) in enumerate(edges):
            self.pending.append((start + k * EDGE_GAP, name, down))

    def _release_hold(self, tick: int) -> None:
        if self.holding:
            self._schedule(tick, [(A.HOLDS[self.holding], False)])
            self.holding = None

    def request(self, action: str, mode: str, tick: int) -> None:
        """一次性动作（手势刚被确认时调用一次）。"""
        if action not in A.ALL_ACTIONS:
            self.note = f"未知动作 {action}"
            return
        if action in A.HOLDS:
            self.hold(action, True, mode, tick)
            return
        self._release_hold(tick)
        self.deferred = None
        if mode == "jump":
            self.note = "跳跃中，不能打断"
            return
        if action == A.STOP:
            if mode in A.LEAVEABLE:
                self._schedule(tick, [("key_left_ctrl", True), ("key_left_ctrl", False)])
                self.note = "停下"
            return
        if action == mode:
            self.note = f"已经在{A.label(action)}"
            return
        if action in A.DANCES:
            if mode == "locomotion" or mode in A.LEAVEABLE:
                key = A.DANCES[action]
                self._schedule(tick, [("key_left_ctrl", True), (key, True),
                                      (key, False), ("key_left_ctrl", False)])
                self.note = A.label(action)
            return
        # 固定动作和跳跃：只能从 locomotion 进入
        if mode == "locomotion":
            key = A.MOTIONS[action][0]
            self._schedule(tick, [(key, True), (key, False)])
            self.note = A.label(action)
        elif mode in A.LEAVEABLE:
            self._schedule(tick, [("key_left_ctrl", True), ("key_left_ctrl", False)])
            self.deferred, self.deferred_since = action, None
            self.note = f"先停下，再{A.label(action)}"

    def hold(self, action: str, on: bool, mode: str, tick: int) -> None:
        """持续动作：手势出现时 on=True，消失时 on=False。"""
        if on:
            if mode != "locomotion" or self.holding == action:
                return
            self._release_hold(tick)
            self._schedule(tick, [(A.HOLDS[action], True)])
            self.holding = action
            self.note = A.label(action)
        elif self.holding == action:
            self._release_hold(tick)

    def due(self, mode: str, tick: int) -> list[tuple[str, bool]]:
        """本 tick 该发出的边沿。每个控制周期调用一次。"""
        if self.holding and mode != "locomotion":
            self._release_hold(tick)
        if self.deferred:
            if mode == "locomotion":
                if self.deferred_since is None:
                    self.deferred_since = tick
                elif tick - self.deferred_since >= SETTLE_TICKS:
                    action, self.deferred = self.deferred, None
                    self.request(action, mode, tick)
            else:
                self.deferred_since = None
        out = [(name, down) for t, name, down in self.pending if t <= tick]
        self.pending = [p for p in self.pending if p[0] > tick]
        return out
