"""跳跳动作表：动作名 → 官方 jumper.app 的按键，以及每个动作能从哪些模式进入。

按键和模式全部来自官方 bundle（`out/bundle_example/jumper.app` 的 controller.toml），
这里只是把它们写成表，方便手势层按名字调用。
"""

from __future__ import annotations

# 固定动作：只能从站立行走（locomotion）进入，播完自动回到 locomotion。
MOTIONS = {
    "gesture_hello": ["key_1"],
    "gesture_bow": ["key_2"],
    "gesture_paw": ["key_3"],
    "gesture_salute": ["key_4"],
    "jump": ["key_space"],
}

# 舞蹈：Ctrl + 数字，可以从 locomotion、任一支舞或任一个固定动作进入。
DANCES = {
    "dance_crab": "key_1",
    "dance_brazilian": "key_2",
    "dance_maze": "key_3",
    "dance_dream_wings": "key_4",
}

# 持续动作：手势保持期间按住对应按键（只在 locomotion 里有效）。
HOLDS = {
    "walk_forward": "key_w",
    "turn_left": "key_j",
    "turn_right": "key_l",
}

STOP = "stop"

# 单按 Ctrl 能打断的模式（跳跃不能被打断，官方设计如此）。
LEAVEABLE = {"gesture_hello", "gesture_bow", "gesture_paw", "gesture_salute", *DANCES}

ALL_ACTIONS = {*MOTIONS, *DANCES, *HOLDS, STOP}

# 界面上显示的中文名
LABELS = {
    "locomotion": "站立 / 行走",
    "gesture_hello": "挥钳打招呼",
    "gesture_bow": "鞠躬",
    "gesture_paw": "伸爪",
    "gesture_salute": "敬礼",
    "jump": "跳跃",
    "dance_crab": "螃蟹舞",
    "dance_brazilian": "巴西舞",
    "dance_maze": "迷宫舞",
    "dance_dream_wings": "梦之翼舞",
    "claw_left": "左钳模式",
    "claw_right": "右钳模式",
    "safe": "保护姿态",
    "walk_forward": "向前走",
    "turn_left": "向左转",
    "turn_right": "向右转",
    "stop": "停下",
}


def label(name: str) -> str:
    return LABELS.get(name, name)
