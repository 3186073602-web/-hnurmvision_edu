#include "io_context.hpp"

#include "vision_gt.hpp"

#include <hnurm_interfaces/msg/vision_send_data.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <chrono>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace training::io {
namespace {

// 第二讲：仿真 IO。
//
// 与仿真器之间的约定（见 simulation/simulator/src/ros2/topic.rs）：
//   输出图像：/image_raw  sensor_msgs/Image（rgb8，1440x1080）
//   接收指令：/vision_send_data  hnurm_interfaces/VisionSendData
// 两路都用 SensorData QoS（best effort），与仿真端的 r2r 配置保持一致。
struct Bridge {
  std::shared_ptr<rclcpp::Node> node;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_subscription;
  rclcpp::Publisher<hnurm_interfaces::msg::VisionSendData>::SharedPtr command_publisher;
  std::mutex mutex;
  cv::Mat latest_image;          // 最近一帧 BGR 图像
  TimePoint latest_stamp{};      // 收到该帧时的本机时刻
  bool has_image = false;
  bool started = false;

  // /tf 里 map 系下的装甲板中心与云台位姿，用于第四讲的真值对比。
  std::map<std::string, Eigen::Vector3d> armor_positions;
  Eigen::Vector3d gimbal_position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond gimbal_rotation = Eigen::Quaterniond::Identity();
  // 底盘（odom）与云台的局部量：云台世界姿态 = 底盘姿态 × 云台局部姿态。
  // 只看 gimbal_link 的局部旋转会在底盘转向后丢掉底盘角，导致 yaw 全错。
  Eigen::Vector3d odom_position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond odom_rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d gimbal_local_position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond gimbal_local_rotation = Eigen::Quaterniond::Identity();
  bool has_odom = false;
  bool has_gimbal = false;
};

Bridge& bridge() {
  static Bridge instance;
  return instance;
}

void onTf(const tf2_msgs::msg::TFMessage::SharedPtr msg) {
  Bridge& b = bridge();
  std::lock_guard<std::mutex> lock(b.mutex);
  for (const auto& tf : msg->transforms) {
    const std::string& child = tf.child_frame_id;
    const auto& t = tf.transform.translation;
    const auto& q = tf.transform.rotation;
    if (child == "odom") {
      // map -> odom：底盘在世界系中的位姿。
      b.odom_position = Eigen::Vector3d(t.x, t.y, t.z);
      b.odom_rotation = Eigen::Quaterniond(q.w, q.x, q.y, q.z);
      b.has_odom = true;
      b.gimbal_position = b.odom_position;  // 兜底：没有 gimbal_link 时用底盘位置
    } else if (child == "gimbal_link") {
      // odom -> gimbal_link：云台相对底盘的局部位姿。
      b.gimbal_local_position = Eigen::Vector3d(t.x, t.y, t.z);
      b.gimbal_local_rotation = Eigen::Quaterniond(q.w, q.x, q.y, q.z);
      b.has_gimbal = true;
    } else if (child.rfind("armor_", 0) == 0) {
      b.armor_positions[child] = Eigen::Vector3d(t.x, t.y, t.z);
    }
  }
}

void onImage(const sensor_msgs::msg::Image::SharedPtr msg) {
  if (!msg || msg->data.empty()) return;
  if (msg->encoding != "rgb8" && msg->encoding != "bgr8") return;

  // 用行跨度构造视图，避免 ROS 图像按 4 字节对齐产生的行尾填充。
  const cv::Mat view(static_cast<int>(msg->height), static_cast<int>(msg->width),
                     CV_8UC3, const_cast<unsigned char*>(msg->data.data()),
                     static_cast<std::size_t>(msg->step));
  cv::Mat decoded;
  if (msg->encoding == "rgb8") {
    cv::cvtColor(view, decoded, cv::COLOR_RGB2BGR);
  } else {
    decoded = view.clone();
  }
  if (decoded.empty()) return;

  Bridge& b = bridge();
  std::lock_guard<std::mutex> lock(b.mutex);
  b.latest_image = std::move(decoded);
  b.latest_stamp = Clock::now();
  b.has_image = true;
}

}  // namespace

std::optional<Eigen::Vector3d> nearestArmorInGimbal(const Eigen::Vector3d& hint) {
  Bridge& b = bridge();
  std::lock_guard<std::mutex> lock(b.mutex);
  if (!b.has_gimbal || b.armor_positions.empty()) return std::nullopt;

  std::optional<Eigen::Vector3d> best;
  double best_error = 0.0;
  for (const auto& [name, world] : b.armor_positions) {
    (void)name;
    const Eigen::Vector3d relative = b.gimbal_rotation.inverse() * (world - b.gimbal_position);
    const double error = (relative - hint).norm();
    if (!best || error < best_error) {
      best = relative;
      best_error = error;
    }
  }
  if (best_error > 1.0) return std::nullopt;  // 最近的板子也差 1m 以上，认为没对上
  return best;
}

void IoContext::open() {
  Bridge& b = bridge();
  if (b.started) return;

  if (!rclcpp::ok()) rclcpp::init(0, nullptr);

  b.node = std::make_shared<rclcpp::Node>("hnurmvision_training");

  // 仿真器以 sensor_data 的 QoS 发布图像；订阅端必须匹配 best effort，
  // 否则在默认 reliable QoS 下收不到任何消息。
  // 仿真以默认的 reliable QoS 发布原图，订阅端保持同样的 QoS。
  b.image_subscription = b.node->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", rclcpp::QoS(10),
      [](const sensor_msgs::msg::Image::SharedPtr msg) { onImage(msg); });

  b.tf_subscription = b.node->create_subscription<tf2_msgs::msg::TFMessage>(
      "/tf", rclcpp::QoS(100),
      [](const tf2_msgs::msg::TFMessage::SharedPtr msg) { onTf(msg); });

  b.command_publisher =
      b.node->create_publisher<hnurm_interfaces::msg::VisionSendData>(
          "/vision_send_data", rclcpp::SensorDataQoS());

  b.started = true;
}

std::optional<SensorData> IoContext::read() {
  Bridge& b = bridge();
  if (!b.started) return std::nullopt;

  // 单线程顺序主循环里没有 executor 线程，这里由主循环驱动回调。
  rclcpp::spin_some(b.node);

  cv::Mat frame;
  TimePoint stamp;
  {
    std::lock_guard<std::mutex> lock(b.mutex);
    if (!b.has_image) return std::nullopt;
    frame = b.latest_image.clone();
    stamp = b.latest_stamp;
  }

  SensorData data;
  data.image = std::move(frame);
  data.timestamp = stamp;
  {
    std::lock_guard<std::mutex> lock(b.mutex);
    if (b.has_gimbal) {
      const Eigen::Quaterniond base = b.has_odom ? b.odom_rotation.normalized()
                                                 : Eigen::Quaterniond::Identity();
      data.imu_q = (base * b.gimbal_local_rotation.normalized()).normalized();
      data.gimbal_origin = b.odom_position + base * b.gimbal_local_position;
    } else {
      data.imu_q = Eigen::Quaterniond::Identity();
      data.gimbal_origin = b.odom_position;
    }
  }
  data.bullet_speed = 25.0;  // 仿真弹速，第四讲预测与第五讲解算使用
  return data;
}

void IoContext::write(const Command& command) {
  Bridge& b = bridge();
  if (!b.started || !b.command_publisher) return;

  hnurm_interfaces::msg::VisionSendData msg;
  msg.header.stamp = b.node->now();
  msg.header.frame_id = "vision";

  // 0 = 丢失目标，1 = 跟随中，2 = 请求发射；仿真端只在非 0 时接管云台。
  msg.target_state.data = command.control ? (command.shoot ? 2 : 1) : 0;
  msg.target_type.data = 0;  // 未分类

  // Command 里的绝对角是弧度（x 前、y 左、z 上的右手系），而串口协议使用
  // RM 约定的角度制、pitch 向上为正。
  msg.yaw = static_cast<float>(command.yaw * 180.0 / kPi);
  msg.pitch = static_cast<float>(command.pitch * 180.0 / kPi);
  msg.delta_pitch = 0.0F;
  msg.delta_yaw = 0.0F;
  msg.target_distance = 0.0F;
  msg.original_yout = msg.yaw;
  msg.vel_x = 0.0F;
  msg.vel_y = 0.0F;
  msg.vel_yaw = 0.0F;
  msg.control_id = 0.0F;

  b.command_publisher->publish(msg);
}

void IoContext::close() {
  Bridge& b = bridge();
  if (!b.started) return;
  b.image_subscription.reset();
  b.command_publisher.reset();
  b.node.reset();
  b.started = false;
  if (rclcpp::ok()) rclcpp::shutdown();
}
}  // namespace training::io
