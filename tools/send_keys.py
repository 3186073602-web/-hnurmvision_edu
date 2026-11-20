#!/usr/bin/env python3
"""在 X11 上按住一组按键若干秒，用于驱动仿真器的底盘/云台。

用法：
    python3 tools/send_keys.py --window daedalus --seconds 2 w q
"""
import argparse
import time

from Xlib import X, XK, display
from Xlib.ext import xtest


def find_window(dpy, needle: str):
    root = dpy.screen().root
    stack = [root]
    while stack:
        win = stack.pop()
        try:
            name = win.get_wm_name()
        except Exception:
            name = None
        if name and needle.lower() in name.lower():
            return win
        try:
            stack.extend(win.query_tree().children)
        except Exception:
            pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", default="", help="窗口标题关键字，空则用当前焦点")
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("keys", nargs="+")
    args = parser.parse_args()

    dpy = display.Display()
    if args.window:
        win = find_window(dpy, args.window)
        if win is None:
            raise SystemExit(f"window not found: {args.window}")
        dpy.set_input_focus(win, X.RevertToParent, X.CurrentTime)
        dpy.sync()
        time.sleep(0.2)

    codes = [dpy.keysym_to_keycode(XK.string_to_keysym(k)) for k in args.keys]
    for code in codes:
        xtest.fake_input(dpy, X.KeyPress, code)
    dpy.sync()
    time.sleep(args.seconds)
    for code in reversed(codes):
        xtest.fake_input(dpy, X.KeyRelease, code)
    dpy.sync()


if __name__ == "__main__":
    main()
