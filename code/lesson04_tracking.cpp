#include "lessons.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace training {
namespace {

// 装甲板物理尺寸（米）。小板 135×55 mm，大板（哨兵）225×55 mm。
// 第三讲的角点已经外扩到整块板，所以直接用板的外沿尺寸。
constexpr double kSmallHalfWidth = 0.0675;
constexpr double kLargeHalfWidth = 0.1125;
constexpr double kHalfHeight = 0.0275;

// 超过这个时间还没更新，就认为目标丢失，速度清零。
constexpr double kMaxTrackGap = 0.5;

std::vector<cv::Point3f> armorObjectPoints(bool large) {
  const double hw = large ? kLargeHalfWidth : kSmallHalfWidth;
  const double hh = kHalfHeight;
  // 与图像角点顺序一致：左上、右上、右下、左下（y 向下为正）。
  return {
      {static_cast<float>(-hw), static_cast<float>(-hh), 0.0F},
      {static_cast<float>(hw), static_cast<float>(-hh), 0.0F},
      {static_cast<float>(hw), static_cast<float>(hh), 0.0F},
      {static_cast<float>(-hw), static_cast<float>(hh), 0.0F},
  };
}

double secondsBetween(TimePoint a, TimePoint b) {
  return std::chrono::duration<double>(a - b).count();
}

// 单块装甲板的 PnP：返回相机系下的平移（板中心）。
bool solveArmorPose(const Armor& armor, const CameraCalibration& calibration,
                    Eigen::Vector3d& camera_xyz) {
  if (!calibration.valid()) return false;
  std::vector<cv::Point2f> image_points(armor.corners.begin(), armor.corners.end());
  cv::Vec3d rvec, tvec;
  try {
    // IPPE 专门用于平面四点，比迭代法稳定，且不需要初值。
    if (!cv::solvePnP(armorObjectPoints(armor.large_armor), image_points,
                      calibration.matrix, calibration.distortion, rvec, tvec, false,
                      cv::SOLVEPNP_IPPE)) {
      return false;
    }
  } catch (const cv::Exception&) {
    return false;
  }
  if (tvec[2] <= 0.0) return false;  // 板子在相机后面，丢弃
  camera_xyz = Eigen::Vector3d(tvec[0], tvec[1], tvec[2]);
  return true;
}

}  // namespace

std::optional<Target> lesson04_tracking(const std::vector<Armor>& armors,
    TimePoint timestamp, const Eigen::Quaterniond& imu_q,
    const CameraCalibration& calibration, TrackingState& state) {
  // ===== 实现区 BEGIN =====
  if (armors.empty()) {
    state.valid = false;
    return std::nullopt;
  }

  // 选板：逐块解 PnP，取离云台最近的一块。仿真里同一时刻只有一块可见，
  // 实车多目标时这里可以换成“威胁优先（编号/血量）+ 距离”的策略。
  bool found = false;
  Eigen::Vector3d best_position = Eigen::Vector3d::Zero();
  double best_distance = 0.0;

  for (const auto& armor : armors) {
    Eigen::Vector3d camera_xyz;
    if (!solveArmorPose(armor, calibration, camera_xyz)) continue;

    // 相机系 → 云台系 → 世界系（IMU 稳定系）。
    const Eigen::Vector3d gimbal_xyz =
        calibration.camera_to_gimbal * camera_xyz + calibration.camera_to_gimbal_translation;
    const Eigen::Vector3d world_xyz = imu_q * gimbal_xyz;

    const double distance = world_xyz.norm();
    if (!found || distance < best_distance) {
      found = true;
      best_distance = distance;
      best_position = world_xyz;
    }
  }

  if (!found) {
    state.valid = false;
    return std::nullopt;
  }

  // 匀速模型：相邻两帧差分估计速度；丢帧太久或时间间隔异常时清零。
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  const double now_sec =
      std::chrono::duration<double>(timestamp.time_since_epoch()).count();
  if (state.valid && state.last_time_sec > 0.0) {
    const double dt_pair = now_sec - state.last_time_sec;
    if (dt_pair > 1e-3 && dt_pair < kMaxTrackGap) {
      // 差分速度抖动大，做一阶低通（新值权重 0.5），预测框才不会来回跳。
      const Eigen::Vector3d measured = (best_position - state.position) / dt_pair;
      velocity = state.velocity * 0.5 + measured * 0.5;
    }
  }

  state.valid = true;
  state.position = best_position;
  state.velocity = velocity;
  state.last_time_sec = now_sec;
  state.last_distance = best_distance;

  Target target;
  target.position = best_position;
  target.velocity = velocity;
  return target;
  // ===== 实现区 END =====
}
}  // namespace training
