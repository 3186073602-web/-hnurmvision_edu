#include "lessons.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace training {
namespace {

// 装甲板物理尺寸（米）。小板 135×55 mm，大板（哨兵）225×55 mm。
// 第三讲的角点已经外扩到整块板，所以直接用板的外沿尺寸。
// 标定系数：仿真渲染的装甲板比标称 135×55 mm 略大（按 /tf 真值反算约 3.3%），
// 乘上它可以把 PnP 的距离偏差从 ~9 cm 压到 ~1 cm 量级。
constexpr double kPlateScale = 1.033;
constexpr double kSmallHalfWidth = 0.0675 * kPlateScale;
constexpr double kLargeHalfWidth = 0.1125 * kPlateScale;
constexpr double kHalfHeight = 0.0275 * kPlateScale;

// 超过这个时间还没更新，就认为目标丢失，速度清零。
constexpr double kMaxTrackGap = 0.5;

// 常加速度卡尔曼滤波参数。
// 过程噪声 q 取“加加速度”谱密度（m²/s⁵），量测噪声 r 取 PnP 位置方差（m²）。
// 目标按 ±0.5 m / 4 s 正弦平移，加速度约 1.2 m/s²，q 取 4 足够跟上；
// r 对应约 1.5 cm 的 PnP 抖动。
constexpr double kProcessNoise = 4.0;
constexpr double kMeasurementNoise = 1.5e-2 * 1.5e-2;
constexpr double kMaxSpeed = 4.0;
constexpr double kMaxAccel = 8.0;

// 单轴常加速度卡尔曼一步预测 + 更新。
void kalmanStep(double measurement, double dt, double& position, double& velocity,
                double& acceleration, Eigen::Matrix3d& covariance) {
  Eigen::Matrix3d transition;
  transition << 1.0, dt, 0.5 * dt * dt,
                0.0, 1.0, dt,
                0.0, 0.0, 1.0;

  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;
  const double dt4 = dt3 * dt;
  const double dt5 = dt4 * dt;
  Eigen::Matrix3d process;
  process << dt5 / 20.0, dt4 / 8.0, dt3 / 6.0,
             dt4 / 8.0, dt3 / 3.0, dt2 / 2.0,
             dt3 / 6.0, dt2 / 2.0, dt;
  process *= kProcessNoise;

  Eigen::Vector3d state(position, velocity, acceleration);
  state = transition * state;
  covariance = transition * covariance * transition.transpose() + process;

  const Eigen::Vector3d observation(1.0, 0.0, 0.0);
  const double innovation_var = observation.dot(covariance * observation) + kMeasurementNoise;
  const Eigen::Vector3d gain = covariance * observation / innovation_var;
  state += gain * (measurement - state(0));
  covariance = (Eigen::Matrix3d::Identity() - gain * observation.transpose()) * covariance;

  position = state(0);
  velocity = state(1);
  acceleration = state(2);
}

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

  // IPPE 只保证平面四点的初值，再用 LM 最小化重投影误差，顺带拒绝离谱解。
  const auto object_points = armorObjectPoints(armor.large_armor);
  try {
    cv::solvePnPRefineLM(object_points, image_points, calibration.matrix,
                         calibration.distortion, rvec, tvec);
  } catch (const cv::Exception&) {
    return false;
  }
  std::vector<cv::Point2f> projected;
  cv::projectPoints(object_points, rvec, tvec, calibration.matrix,
                    calibration.distortion, projected);
  double squared = 0.0;
  for (std::size_t i = 0; i < projected.size(); ++i) {
    const double dx = projected[i].x - image_points[i].x;
    const double dy = projected[i].y - image_points[i].y;
    squared += dx * dx + dy * dy;
  }
  const double rmse = std::sqrt(squared / projected.size());
  if (rmse > 15.0) return false;      // 重投影误差过大，视为坏解

  camera_xyz = Eigen::Vector3d(tvec[0], tvec[1], tvec[2]);
  return true;
}

}  // namespace

std::optional<Target> lesson04_tracking(const std::vector<Armor>& armors,
    TimePoint timestamp, const Eigen::Quaterniond& imu_q,
    const Eigen::Vector3d& world_origin, const CameraCalibration& calibration,
    TrackingState& state) {
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

  // 已锁定时选“离上一帧位置最近”的板，避免在多块板之间来回跳；
  // 首次锁定或跳得太远时退回“离相机最近”。
  const bool tracking = state.valid;
  double best_score = 0.0;
  for (const auto& armor : armors) {
    Eigen::Vector3d camera_xyz;
    if (!solveArmorPose(armor, calibration, camera_xyz)) continue;

    const Eigen::Vector3d gimbal_xyz =
        calibration.camera_to_gimbal * camera_xyz + calibration.camera_to_gimbal_translation;
    const Eigen::Vector3d world_xyz = imu_q * gimbal_xyz + world_origin;

    const double distance = world_xyz.norm();
    const double score = tracking ? (world_xyz - state.position).norm() : distance;
    if (!found || score < best_score) {
      found = true;
      best_score = score;
      best_distance = distance;
      best_position = world_xyz;
      state.last_camera_xyz = camera_xyz;
    }
  }
  if (tracking && found && best_score > 0.8) {
    // 跳变过大：当作重新捕获，用最近的板。
    found = false;
    for (const auto& armor : armors) {
      Eigen::Vector3d camera_xyz;
      if (!solveArmorPose(armor, calibration, camera_xyz)) continue;
      const Eigen::Vector3d gimbal_xyz =
          calibration.camera_to_gimbal * camera_xyz + calibration.camera_to_gimbal_translation;
      const Eigen::Vector3d world_xyz = imu_q * gimbal_xyz + world_origin;
      const double distance = world_xyz.norm();
      if (!found || distance < best_distance) {
        found = true;
        best_distance = distance;
        best_position = world_xyz;
        state.last_camera_xyz = camera_xyz;
      }
    }
  }

  if (!found) {
    // 短时丢失：保持滤波状态，用上一帧速度和加速度外推，最多顶 150 ms。
    // 超过之后才判定目标丢失，避免一帧漏检就把状态清空、云台跟着乱跳。
    const double now = std::chrono::duration<double>(timestamp.time_since_epoch()).count();
    const double gap = state.last_time_sec > 0.0 ? now - state.last_time_sec : 1e9;
    if (state.valid && gap < 0.15 && ++state.miss_frames <= 8) {
      Target held;
      held.position = state.position + state.velocity * gap +
                      0.5 * state.acceleration * gap * gap - world_origin;
      held.velocity = state.velocity;
      held.acceleration = state.acceleration;
      return held;
    }
    state.valid = false;
    state.miss_frames = 0;
    return std::nullopt;
  }
  state.miss_frames = 0;

  // 常加速度卡尔曼：先用上一帧状态预测，再用 PnP 位置更新。
  Eigen::Vector3d position = best_position;
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
  const double now_sec =
      std::chrono::duration<double>(timestamp.time_since_epoch()).count();
  if (state.valid && state.last_time_sec > 0.0) {
    const double dt_pair = now_sec - state.last_time_sec;
    if (dt_pair > 1e-3 && dt_pair < kMaxTrackGap) {
      position = state.position;
      velocity = state.velocity;
      acceleration = state.acceleration;
      for (int axis = 0; axis < 3; ++axis) {
        double p = position(axis), v = velocity(axis), a = acceleration(axis);
        kalmanStep(best_position(axis), dt_pair, p, v, a, state.covariance[axis]);
        position(axis) = p;
        velocity(axis) = v;
        acceleration(axis) = a;
      }
      const double speed = velocity.norm();
      if (speed > kMaxSpeed) velocity *= kMaxSpeed / speed;
      if (acceleration.norm() > kMaxAccel) {
        acceleration *= kMaxAccel / acceleration.norm();
      }
    } else {
      for (auto& covariance : state.covariance) {
        covariance = Eigen::Matrix3d::Identity() * 1.0;
      }
    }
  } else {
    // 初始化：位置直接取观测，速度和加速度置零，协方差给足。
    for (auto& covariance : state.covariance) {
      covariance = Eigen::Matrix3d::Zero();
      covariance(0, 0) = 1.0;
      covariance(1, 1) = 10.0;
      covariance(2, 2) = 10.0;
    }
  }

  state.valid = true;
  state.position = position;
  state.velocity = velocity;
  state.acceleration = acceleration;
  state.last_time_sec = now_sec;
  state.last_distance = best_distance;

  // 课程约定 Target.position 是“相对云台原点”的向量；滤波在 map 系里做，
  // 输出时减去当前云台原点，得到 map 朝向、云台中心的相对位置。
  Target target;
  target.position = position - world_origin;
  target.velocity = velocity;
  target.acceleration = acceleration;
  return target;
  // ===== 实现区 END =====
}
}  // namespace training
