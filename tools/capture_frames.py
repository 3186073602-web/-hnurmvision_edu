#!/usr/bin/env python3
"""从仿真器订阅 /image_raw，按固定间隔保存相机原始帧。

用法：
    source /opt/ros/humble/setup.bash
    export ROS_DOMAIN_ID=89
    python3 tools/capture_frames.py <输出目录> [最大帧数] [间隔帧数]
"""
import os
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class FrameSaver(Node):
    def __init__(self, out_dir: str, limit: int, stride: int) -> None:
        super().__init__("hnurm_frame_saver")
        self.out_dir = out_dir
        self.limit = limit
        self.stride = stride
        self.seen = 0
        self.saved = 0
        os.makedirs(out_dir, exist_ok=True)
        # 仿真以默认 reliable QoS 发布 /image_raw。
        self.sub = self.create_subscription(Image, "/image_raw", self.on_image, 10)
        self.get_logger().info(f"saving frames to {out_dir}")

    def on_image(self, msg: Image) -> None:
        self.seen += 1
        if self.saved >= self.limit:
            return
        if self.seen % self.stride != 0:
            return
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        row = buf.reshape(msg.height, msg.step)
        frame = row[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path = os.path.join(self.out_dir, f"frame_{self.saved:05d}.png")
        cv2.imwrite(path, frame)
        self.saved += 1
        if self.saved % 50 == 0:
            self.get_logger().info(f"saved {self.saved} frames")


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset/raw"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    rclpy.init()
    node = FrameSaver(out_dir, limit, stride)
    try:
        while rclpy.ok() and node.saved < limit:
            rclpy.spin_once(node, timeout_sec=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"done: {node.saved} frames")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
