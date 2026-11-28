#pragma once
#include <Eigen/Geometry>
#include <optional>

namespace training::io {
// 从仿真器发布的 /tf 中取“离 hint 最近的装甲板中心”，换算到云台坐标系（m）。
// 仅用于第四讲的误差对比：hint 传 PnP 解算结果，返回同一块板子的真实位置。
// 没有 /tf、或还没有装甲板帧时返回 nullopt。
std::optional<Eigen::Vector3d> nearestArmorInGimbal(const Eigen::Vector3d& hint);
// 同上，但返回 map 系下的坐标（标定用）。
std::optional<Eigen::Vector3d> nearestArmorInWorld(const Eigen::Vector3d& hint);
}  // namespace training::io
