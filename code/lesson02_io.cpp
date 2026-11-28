#include "io_context.hpp"

#include "vision_gt.hpp"

#include <hnurm_interfaces/msg/vision_send_data.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <chrono>
#include <deque>
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
  // 图像帧号与时间戳：用来发现重复帧、对齐 TF。
  uint64_t latest_image_id = 0;
  uint64_t consumed_image_id = 0;
  double latest_image_stamp = 0.0;
  struct PoseSample {
    double stamp = 0.0;   // ROS 时间戳（诊断用）
    double recv = 0.0;    // 接收时刻（steady 时钟，用于和图像对齐）
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();       // 云台世界姿态
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();                    // 云台世界位置
    Eigen::Quaterniond camera_rotation = Eigen::Quaterniond::Identity(); // 相机世界姿态
    Eigen::Vector3d camera_origin = Eigen::Vector3d::Zero();             // 相机世界位置
    bool has_camera = false;
  };
  // 相机链的局部量：gimbal_link -> camera_link -> camera_optical_frame
  Eigen::Vector3d camera_link_translation = Eigen::Vector3d::Zero();
  Eigen::Quaterniond camera_link_rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d optical_translation = Eigen::Vector3d::Zero();
  Eigen::Quaterniond optical_rotation = Eigen::Quaterniond::Identity();
  bool has_camera_chain = false;
  std::deque<PoseSample> pose_samples;
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
      if (b.has_odom) {
        const Eigen::Quaterniond base = b.odom_rotation.normalized();
        Bridge::PoseSample sample;
        sample.stamp = static_cast<double>(tf.header.stamp.sec) +
                       static_cast<double>(tf.header.stamp.nanosec) * 1e-9;
        sample.recv = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        sample.rotation = (base * b.gimbal_local_rotation.normalized()).normalized();
        sample.origin = b.odom_position + base * b.gimbal_local_position;
        if (b.has_camera_chain) {
          // 相机世界位姿 = 云台世界位姿 × (camera_link × optical)
          const Eigen::Quaterniond rel_rotation =
              (b.camera_link_rotation * b.optical_rotation).normalized();
          const Eigen::Vector3d rel_translation =
              b.camera_link_translation + b.camera_link_rotation * b.optical_translation;
          sample.camera_rotation = (sample.rotation * rel_rotation).normalized();
          sample.camera_origin = sample.origin + sample.rotation * rel_translation;
          sample.has_camera = true;
        }
        b.pose_samples.push_back(sample);
        while (!b.pose_samples.empty() &&
               sample.stamp - b.pose_samples.front().stamp > 1.0) {
          b.pose_samples.pop_front();
        }
      }
    } else if (child == "camera_link") {
      b.camera_link_translation = Eigen::Vector3d(t.x, t.y, t.z);
      b.camera_link_rotation = Eigen::Quaterniond(q.w, q.x, q.y, q.z);
    } else if (child == "camera_optical_frame") {
      b.optical_translation = Eigen::Vector3d(t.x, t.y, t.z);
      b.optical_rotation = Eigen::Quaterniond(q.w, q.x, q.y, q.z);
      b.has_camera_chain = true;
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
  ++b.latest_image_id;
  b.latest_image_stamp = static_cast<double>(msg->header.stamp.sec) +
                         static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
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
  uint64_t image_id = 0;
  double image_stamp = 0.0;
  {
    std::lock_guard<std::mutex> lock(b.mutex);
    if (!b.has_image) return std::nullopt;
    // 没有新帧就直接返回，避免同一帧被处理两次。
    if (b.latest_image_id == b.consumed_image_id) return std::nullopt;
    b.consumed_image_id = b.latest_image_id;
    frame = b.latest_image.clone();
    stamp = b.latest_stamp;
    image_id = b.latest_image_id;
    image_stamp = b.latest_image_stamp;
  }

  SensorData data;
  data.image = std::move(frame);
  data.timestamp = stamp;
  data.image_id = image_id;
  data.image_stamp = image_stamp;
  {
    std::lock_guard<std::mutex> lock(b.mutex);
    // 优先用与图像采集时刻最接近的 TF 样本（50 ms 内），否则退回最新值。
    const double image_recv = std::chrono::duration<double>(stamp.time_since_epoch()).count();
    const Bridge::PoseSample* best = nullptr;
    double best_dt = 1e9;
    for (const auto& sample : b.pose_samples) {
      const double dt = std::abs(sample.recv - image_recv);
      if (dt < best_dt) {
        best_dt = dt;
        best = &sample;
      }
    }
    if (best != nullptr && best_dt <= 0.03) {
      data.imu_q = best->rotation;
      data.gimbal_origin = best->origin;
      data.tf_stamp = best->stamp;
      data.tf_sync_ms = (best->recv - image_recv) * 1000.0;
      data.tf_interpolated = best_dt > 0.01;
      if (best->has_camera) {
        data.camera_to_world = best->camera_rotation;
        data.camera_origin = best->camera_origin;
        data.has_camera_pose = true;
      }
    } else if (b.has_gimbal) {
      const Eigen::Quaterniond base = b.has_odom ? b.odom_rotation.normalized()
                                                 : Eigen::Quaterniond::Identity();
      data.imu_q = (base * b.gimbal_local_rotation.normalized()).normalized();
      data.gimbal_origin = b.odom_position + base * b.gimbal_local_position;
      data.tf_interpolated = true;
    } else {
      data.imu_q = Eigen::Quaterniond::Identity();
      data.gimbal_origin = b.odom_position;
      data.tf_interpolated = true;
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
