#include "lessons.hpp"

#include <cmath>
#include <chrono>

namespace training {
// 第五讲：弹道解算
//
// 输入格式：
//   target       : const Target&；position 单位 m，velocity 单位 m/s，二者均采用
//                  第四讲规定的 x 向前、y 向左、z 向上坐标系。
//   timestamp    : TimePoint，target 对应的图像时刻，用于计算预测时间。
//   bullet_speed : double，弹丸初速度，单位 m/s，必须为有限正数。
// 输出格式：
//   Command：
//   - control : true 表示请求云台跟随；无法产生安全解时必须为 false；
//   - shoot   : true 表示请求发射，最终仍受 IO 层 allow_fire 和有效性门控；
//   - yaw     : 世界/IMU 参考系下的绝对偏航角，单位 rad；
//   - pitch   : 世界/IMU 参考系下的绝对俯仰角，单位 rad。
// 验收：仿真器完成静止靶、平移靶任务。
namespace {
constexpr double kGravity = 9.81;          // m/s²
constexpr double kMountElevation = 27.0;  // 仿真实测零位（课程文档写 25°，实测 27° 命中）
constexpr double kMaxIterations = 5;
constexpr double kSpeedTolerance = 1e-3;   // 飞行时间收敛阈值，s

// 弹道解算：已知水平距离 d、相对高度 h、弹速 v0，返回仰角 theta、飞行时间 tf。
// 解 a·tan²θ − d·tanθ + (a+h) = 0（a = g·d²/(2v0²)），取低伸弹道（时间短的解）。
bool solveBallistics(double d, double h, double v0, double& theta, double& tf) {
  if (!(v0 > 1e-3) || d < 1e-3) return false;
  const double a = kGravity * d * d / (2.0 * v0 * v0);
  const double discriminant = d * d - 4.0 * a * (a + h);
  if (discriminant < 0.0) return false;  // 打不到：太远/太高/弹速太低

  const double tan_theta = (d - std::sqrt(discriminant)) / (2.0 * a);
  theta = std::atan(tan_theta);
  const double cos_theta = std::cos(theta);
  if (std::abs(cos_theta) < 1e-6) return false;
  tf = d / (v0 * cos_theta);
  return std::isfinite(tf) && tf > 0.0;
}
}  // namespace

Command lesson05_aiming(const Target& target, TimePoint timestamp, double bullet_speed) {
  Command command;

  // 处理延迟：观测时刻 → 本函数被调用。仿真没有发弹延时（delay = 0）。
  const double dt1 = std::chrono::duration<double>(Clock::now() - timestamp).count();
  const double delay = 0.0;

  // 不动点迭代：p_hit 依赖 t_f，t_f 又依赖 p_hit。
  double tf = 0.0;
  Eigen::Vector3d hit = target.position;
  double theta = 0.0;
  bool solvable = false;
  for (int i = 0; i < kMaxIterations; ++i) {
    hit = target.position + target.velocity * (dt1 + delay + tf);
    const double d = std::hypot(hit.x(), hit.y());  // 水平距离，不是直线距离
    const double h = hit.z();
    double new_tf = 0.0;
    double new_theta = 0.0;
    if (!solveBallistics(d, h, bullet_speed, new_theta, new_tf)) {
      solvable = false;
      break;
    }
    solvable = true;
    theta = new_theta;
    if (std::abs(new_tf - tf) < kSpeedTolerance) {
      tf = new_tf;
      break;
    }
    tf = new_tf;
  }
  if (!solvable) return command;

  // yaw 用几何：世界系绝对角；pitch 用弹道：25° 零位 − 仰角。
  command.control = true;
  command.yaw = std::atan2(hit.y(), hit.x());
  command.pitch = kMountElevation * kPi / 180.0 - theta;
  return command;
}
}  // namespace training
