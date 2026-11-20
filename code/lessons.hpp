#pragma once
#include "types.hpp"
#include "tracking_state.hpp"
#include <optional>
#include <vector>

namespace training {
std::vector<Armor> lesson03_detection(const cv::Mat& image);

// 能量机关（能量装置）观测：外接框四角 + 置信度。没有机关时返回 nullopt。
struct Rune {
  std::array<cv::Point2f, 4> corners{};  // 左上、右上、右下、左下
  double confidence = 0;
};
std::optional<Rune> lesson03_rune_detection(const cv::Mat& image);
std::optional<Target> lesson04_tracking(const std::vector<Armor>& armors,
    TimePoint timestamp, const Eigen::Quaterniond& imu_q, const CameraCalibration& calibration, TrackingState& state);
Command lesson05_aiming(const Target& target, TimePoint timestamp, double bullet_speed);
}  // namespace training
