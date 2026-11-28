#pragma once
#include "types.hpp"
#include "tracking_state.hpp"
#include <optional>
#include <vector>

namespace training {
std::vector<Armor> lesson03_detection(const cv::Mat& image);

// 能量机关（能量装置）观测：每个圆形小装置一个紧贴的小框 + 置信度。
struct Rune {
  std::array<cv::Point2f, 4> corners{};  // 左上、右上、右下、左下
  double confidence = 0;
};
std::vector<Rune> lesson03_rune_detection(const cv::Mat& image);
std::optional<Target> lesson04_tracking(const std::vector<Armor>& armors,
    TimePoint timestamp, const Eigen::Quaterniond& imu_q,
    const Eigen::Vector3d& world_origin, const Eigen::Quaterniond& camera_to_world,
    const Eigen::Vector3d& camera_origin, const CameraCalibration& calibration,
    TrackingState& state);
Command lesson05_aiming(const Target& target, TimePoint timestamp, double bullet_speed);
}  // namespace training
