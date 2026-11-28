#pragma once
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <array>
#include <chrono>

namespace training {
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
constexpr double kPi = 3.14159265358979323846;

// IO 提供的算法输入；错误、连接状态和设备模式留在 IO 层。
struct CameraCalibration {
  cv::Mat matrix; // 3x3 CV_64F，相机光学坐标系
  cv::Mat distortion;
  Eigen::Matrix3d camera_to_gimbal = Eigen::Matrix3d::Identity();
  Eigen::Vector3d camera_to_gimbal_translation = Eigen::Vector3d::Zero();
  unsigned width=0, height=0;
  bool valid() const { return !matrix.empty(); }
};

struct SensorData {
  cv::Mat image;  // CV_8UC3，BGR，独立拥有像素内存
  TimePoint timestamp{};
  Eigen::Quaterniond imu_q = Eigen::Quaterniond::Identity();
  Eigen::Vector3d gimbal_origin = Eigen::Vector3d::Zero();
  double bullet_speed = 0;  // m/s

  // —— 数据同步诊断（不参与算法，只用于排查时间/重复帧问题）——
  uint64_t image_id = 0;       // 每帧唯一，用来发现重复处理
  double image_stamp = 0.0;    // 图像在 ROS 时钟下的采集时间，s
  double tf_stamp = 0.0;       // 所用 TF 样本的 ROS 时间戳，s
  double tf_sync_ms = 0.0;     // TF 与图像在“接收时刻”上的差，ms（同一 steady 时钟）
  bool tf_interpolated = false;// 是否用了“最近邻”而不是严格同刻 TF
  // 相机光学系在世界系中的位姿（由 TF 链直接给出，不依赖相机→云台外参）。
  Eigen::Quaterniond camera_to_world = Eigen::Quaterniond::Identity();
  Eigen::Vector3d camera_origin = Eigen::Vector3d::Zero();
  bool has_camera_pose = false;
};

// 第三讲输出：一块装甲板的图像观测。
struct Armor {
  std::array<cv::Point2f, 4> corners{};  // 左上、右上、右下、左下
  bool large_armor = false;
  int class_id = -1;
  double confidence = 0;
};

// 第四讲输出：当前观测时刻的单目标状态。无目标用 std::nullopt 表示。
struct Target {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();  // m，IMU稳定系中相对云台原点的目标向量
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();  // m/s，同一坐标系
  // 第四讲滤波给出的加速度估计（m/s²）。用于 250ms 预测的二阶项，
  // 匀速模型在速度反向时会漏掉 1/2·a·t² 这一项。
  Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
};

// 第五讲输出：绝对角使用 rad，IO 边界统一转换单位和方向。
struct Command {
  bool control = false;
  bool shoot = false;
  double yaw = 0;
  double pitch = 0;
};
}  // namespace training
