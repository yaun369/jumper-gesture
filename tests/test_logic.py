"""不依赖仿真和摄像头的逻辑测试：python -m pytest tests -q"""

from jumper_gesture.driver import EDGE_GAP, SETTLE_TICKS, Driver
from jumper_gesture.gestures import Detection, Stabilizer

MAPPING = {
    "palm": {"action": "gesture_hello", "label": "张开手掌"},
    "peace": {"action": "dance_crab", "label": "比耶"},
    "point": {"action": "walk_forward", "label": "食指指向"},
}


def drain(d: Driver, mode: str, start: int, n: int = 40) -> list:
    out = []
    for t in range(start, start + n):
        out += [(t, *e) for e in d.due(mode, t)]
    return out


def test_motion_from_locomotion_is_a_key_tap():
    d = Driver()
    d.request("gesture_hello", "locomotion", 0)
    edges = drain(d, "locomotion", 0)
    assert [(n, down) for _, n, down in edges] == [("key_1", True), ("key_1", False)]
    assert edges[1][0] - edges[0][0] == EDGE_GAP


def test_dance_is_ctrl_chord():
    d = Driver()
    d.request("dance_crab", "locomotion", 0)
    keys = [(n, down) for _, n, down in drain(d, "locomotion", 0)]
    assert keys == [("key_left_ctrl", True), ("key_1", True), ("key_1", False), ("key_left_ctrl", False)]


def test_motion_during_dance_leaves_first_then_fires():
    d = Driver()
    d.request("gesture_hello", "dance_crab", 0)
    first = [(n, down) for _, n, down in drain(d, "dance_crab", 0, 10)]
    assert first == [("key_left_ctrl", True), ("key_left_ctrl", False)]
    later = drain(d, "locomotion", 10, SETTLE_TICKS + 10)
    assert [(n, down) for _, n, down in later] == [("key_1", True), ("key_1", False)]


def test_jump_cannot_be_interrupted_and_repeat_is_ignored():
    d = Driver()
    d.request("gesture_hello", "jump", 0)
    d.request("gesture_hello", "gesture_hello", 1)  # 已经在挥手，不能再按（toggle 会退出）
    assert drain(d, "jump", 0) == []


def test_hold_releases_when_mode_changes():
    d = Driver()
    d.hold("walk_forward", True, "locomotion", 0)
    assert [(n, down) for _, n, down in drain(d, "locomotion", 0, 5)] == [("key_w", True)]
    assert [(n, down) for _, n, down in drain(d, "jump", 5, 10)] == [("key_w", False)]


def det(name, conf=0.9):
    return [Detection(name, conf, (0, 0, 10, 10))]


def test_stabilizer_needs_stable_frames_and_release():
    s = Stabilizer(MAPPING, stable_frames=3, release_frames=2)
    fired = []
    for frame in [det("palm"), det("palm"), det("palm"), det("palm"), [], [], det("palm"),
                  det("palm"), det("palm")]:
        _, ev = s.update(frame)
        fired += [e.kind for e in ev]
    assert fired == ["trigger", "trigger"]  # 举着不重复触发，放下再举才触发第二次


def test_stabilizer_ignores_flicker_and_unmapped():
    s = Stabilizer(MAPPING, stable_frames=3, release_frames=2)
    fired = []
    for frame in [det("palm"), det("peace"), det("palm"), det("fist"), det("fist"), det("fist")]:
        fired += s.update(frame)[1]
    assert fired == []


def test_hold_gesture_emits_on_and_off():
    s = Stabilizer(MAPPING, stable_frames=2, release_frames=2)
    kinds = []
    for frame in [det("point"), det("point"), det("point"), [], []]:
        kinds += [e.kind for e in s.update(frame)[1]]
    assert kinds == ["hold_on", "hold_off"]
