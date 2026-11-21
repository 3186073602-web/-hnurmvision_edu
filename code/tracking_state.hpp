#pragma once
#include <Eigen/Geometry>
#include <optional>

namespace training {
// 仅第四讲使用；由学员设计跨帧状态，其余讲不接收它。
//
// 基础跟踪的状态：上一帧的观测（世界系位置、时间、像素中心）与差分得到的
// 速度估计。帧间丢失时保留上一帧位置，超过 kMaxTrackGap 秒未更新则失效。
struct TrackingState {
  bool valid = false;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();  // m，世界系
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();  // m/s，世界系
  double last_time_sec = 0.0;
  double last_distance = 0.0;                          // m，PnP 距离，用于选板
};
}  // namespace training
