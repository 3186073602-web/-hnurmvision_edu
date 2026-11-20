#!/usr/bin/env python3
"""演示录制时驱动仿真器：先切掉自动瞄准，再按脚本移动底盘和云台。

用法：
    python3 tools/demo_drive.py
"""
import time

from Xlib import X, XK, display as xdisplay
from Xlib.ext import xtest

SCHEDULE = [
    # 小幅、连续地移动底盘和云台，保证装甲板始终在画面里
    ("d", 0.9), ("a", 0.9),
    ("q", 0.6), ("e", 0.6),
    ("Left", 0.35), ("Right", 0.75),
    ("w", 1.0), ("s", 1.0),
    ("Up", 0.20), ("Down", 0.40),
    ("d", 0.9), ("a", 0.9),
    ("Right", 0.4), ("Left", 0.8),
    ("w", 1.1), ("s", 1.1),
    ("q", 0.7), ("e", 0.7),
    ("Left", 0.35), ("Right", 0.7),
    ("d", 0.8), ("a", 0.8),
    ("w", 1.0),
]


def find(window, needle):
    try:
        name = window.get_wm_name()
    except Exception:
        name = None
    if name and needle in name:
        return window
    try:
        for child in window.query_tree().children:
            found = find(child, needle)
            if found:
                return found
    except Exception:
        pass
    return None


def main() -> None:
    dpy = xdisplay.Display(":0")
    root = dpy.screen().root
    sim = find(root, "daedalus")
    if sim is None:
        raise SystemExit("simulator window not found")

    dpy.set_input_focus(sim, X.RevertToParent, X.CurrentTime)
    dpy.sync()
    time.sleep(0.6)

    def tap(name: str) -> None:
        code = dpy.keysym_to_keycode(XK.string_to_keysym(name))
        xtest.fake_input(dpy, X.KeyPress, code)
        dpy.sync()
        time.sleep(0.06)
        xtest.fake_input(dpy, X.KeyRelease, code)
        dpy.sync()
        time.sleep(0.12)

    def hold(name: str, seconds: float) -> None:
        code = dpy.keysym_to_keycode(XK.string_to_keysym(name))
        xtest.fake_input(dpy, X.KeyPress, code)
        dpy.sync()
        time.sleep(seconds)
        xtest.fake_input(dpy, X.KeyRelease, code)
        dpy.sync()
        time.sleep(0.08)

    # F5 关闭“订阅自瞄”，把云台交还键盘，否则方向键不会生效。
    tap("F5")
    time.sleep(1.0)

    for key, seconds in SCHEDULE:
        hold(key, seconds)
    print("drive done")


if __name__ == "__main__":
    main()
