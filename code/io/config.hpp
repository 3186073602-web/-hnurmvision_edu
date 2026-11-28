#pragma once
#include "types.hpp"
#include <array>
#include <string>
namespace training {
struct Config {
  bool display = true;
  bool allow_fire = false;
  // 是否启用能量机关检测（默认关闭：前两项考核只打装甲板，省算力也避免干扰）。
  bool enable_rune = false;
  // 是否画 250ms 预测框与预测误差（调试用，默认关闭）。
  bool show_prediction = false;
  // 独立位姿测量调试模式：打印帧号、采集/处理延迟、TF 对齐误差、PnP 位置。
  bool show_diagnostics = false;
  // 独立位姿测量模式：只算 PnP/跟踪，不向云台发任何指令（用于评估位姿稳定性）。
  bool pose_only = false;
  // 装甲板检测的最低置信度（可视化里的 SMALL 0.xx 那个数）。
  double min_armor_confidence = 0.62;
  std::array<double, 9> camera_matrix{
      1303.6753236814714, 0, 720, 0, 1303.6753236814714, 540, 0, 0, 1};
  std::array<double, 5> distort_coeffs{0, 0, 0, 0, 0};
  std::array<double, 9> camera_to_gimbal{
      0, 0.4226182617, 0.9063077870, -1, 0, 0,
      0, -0.9063077870, 0.4226182617};
  std::array<double, 3> camera_to_gimbal_translation{0, 0, 0};
};
Config load_config(const std::string& path);
CameraCalibration load_calibration(const Config& config);
}
