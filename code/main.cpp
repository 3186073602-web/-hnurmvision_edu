#include "lessons.hpp"
#include "io_context.hpp"
#include "vision_gt.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <deque>
#include <optional>
#include <csignal>
#include <algorithm>
#include <cstdio>
#include <string>

namespace {
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }

// 把第三讲的检测结果画到显示画面上。检测本身在 lesson03_detection 里完成，
// 这里只负责可视化：角点顺序为左上、右上、右下、左下，与 types.hpp 的约定一致。
// 把世界系（IMU 稳定系）下的点投影回图像：世界→云台→相机→像素。
std::optional<cv::Point2f> projectWorld(const Eigen::Vector3d& world,
                                        const Eigen::Quaterniond& imu_q,
                                        const Eigen::Vector3d& origin,
                                        const training::CameraCalibration& calibration) {
  if (!calibration.valid()) return std::nullopt;
  const Eigen::Vector3d gimbal = imu_q.inverse() * (world - origin);
  const Eigen::Vector3d camera =
      calibration.camera_to_gimbal.transpose() * (gimbal - calibration.camera_to_gimbal_translation);
  if (camera.z() <= 1e-3) return std::nullopt;
  const double fx = calibration.matrix.at<double>(0, 0);
  const double fy = calibration.matrix.at<double>(1, 1);
  const double cx = calibration.matrix.at<double>(0, 2);
  const double cy = calibration.matrix.at<double>(1, 2);
  return cv::Point2f(static_cast<float>(fx * camera.x() / camera.z() + cx),
                     static_cast<float>(fy * camera.y() / camera.z() + cy));
}

std::optional<cv::Point2f> projectWorldCamera(const Eigen::Vector3d& world,
                                              const Eigen::Quaterniond& camera_to_world,
                                              const Eigen::Vector3d& camera_origin,
                                              const training::CameraCalibration& calibration) {
  if (!calibration.valid()) return std::nullopt;
  const Eigen::Vector3d camera = camera_to_world.inverse() * (world - camera_origin);
  if (camera.z() < 0.05) return std::nullopt;
  std::vector<cv::Point3f> points{cv::Point3f(static_cast<float>(camera.x()),
                                              static_cast<float>(camera.y()),
                                              static_cast<float>(camera.z()))};
  std::vector<cv::Point2f> image;
  cv::projectPoints(points, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0),
                    calibration.matrix, calibration.distortion, image);
  return image.front();
}

void drawRune(cv::Mat& frame, const training::Rune& rune) {
  cv::Point polygon[4];
  for (int i = 0; i < 4; ++i) {
    polygon[i] = cv::Point(cvRound(rune.corners[i].x), cvRound(rune.corners[i].y));
  }
  for (int i = 0; i < 4; ++i) {
    cv::line(frame, polygon[i], polygon[(i + 1) % 4], cv::Scalar(0, 235, 255), 2, cv::LINE_AA);
  }
  char label[48];
  std::snprintf(label, sizeof(label), "RUNE %.2f", rune.confidence);
  cv::putText(frame, label, cv::Point(polygon[0].x, std::max(14, polygon[0].y - 6)),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 235, 255), 1, cv::LINE_AA);
}

void drawArmors(cv::Mat& frame, const std::vector<training::Armor>& armors) {
  // 只影响显示：检测角点是灯条外沿（P nP 用的物理尺寸），画框时外扩到整块板。
  constexpr float kWiden = 1.35F;
  constexpr float kTall = 2.05F;
  for (const auto& armor : armors) {
    cv::Point2f center(0, 0);
    for (const auto& corner : armor.corners) center += corner;
    center *= 0.25F;

    // corners 是 Point2f，必须逐个取整，不能把 float 直接按 int 重解释。
    cv::Point polygon[4];
    for (int i = 0; i < 4; ++i) {
      const cv::Point2f p(center.x + (armor.corners[i].x - center.x) * kWiden,
                          center.y + (armor.corners[i].y - center.y) * kTall);
      polygon[i] = cv::Point(cvRound(p.x), cvRound(p.y));
    }

    // 大小板用不同颜色区分：绿色 = 小装甲板，橙色 = 大装甲板（哨兵）。
    const cv::Scalar color = armor.large_armor ? cv::Scalar(0, 165, 255)
                                               : cv::Scalar(0, 255, 0);
    for (int i = 0; i < 4; ++i) {
      cv::line(frame, polygon[i], polygon[(i + 1) % 4], color, 2, cv::LINE_AA);
    }

    // 每个角点单独标出来，方便检查角点顺序是否正确。
    for (int i = 0; i < 4; ++i) {
      cv::circle(frame, polygon[i], 3, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
    }

    char label[64];
    std::snprintf(label, sizeof(label), "%s %.2f",
                  armor.large_armor ? "LARGE" : "SMALL", armor.confidence);

    const cv::Point text_anchor{polygon[0].x, std::max(0, polygon[0].y - 8)};
    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
    cv::rectangle(frame,
                  cv::Rect(text_anchor.x, text_anchor.y - text_size.height - 6,
                           text_size.width + 8, text_size.height + 10),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, label,
                cv::Point(text_anchor.x + 4, text_anchor.y - 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
  }

  if (armors.empty()) {
    cv::putText(frame, "NO ARMOR", cv::Point(30, 50),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2,
                cv::LINE_AA);
  }
}
}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  const auto config = training::load_config(argc > 1 ? argv[1] : "configs/simulation.yaml");
  training::setMinArmorConfidence(static_cast<float>(config.min_armor_confidence));
  training::setKeepBestArmor(config.keep_best_armor);
  const auto calibration = training::load_calibration(config);
  training::io::IoContext devices{config};
  training::TrackingState tracking;
  devices.open();

  // 可选：第二个参数是录制文件路径。直接写显示画面，避免窗口被遮挡/缩放
  // 影响录屏质量；30fps、原始分辨率。
  cv::VideoWriter recorder;
  const bool recording = argc > 2;
  if (recording) {
    recorder.open(argv[2], cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0,
                 cv::Size(1440, 1080));
    if (!recorder.isOpened()) {
      std::fprintf(stderr, "cannot open video file %s\n", argv[2]);
      return 1;
    }
  }

  if (config.display) {
    cv::namedWindow("RoboMaster training", cv::WINDOW_AUTOSIZE);
  }
  bool display_configured = false;

  while (!interrupted) {
    auto input = devices.read();
    training::Command command{};
    if (input) {
      const auto armors = training::lesson03_detection(input->image);
      const auto runes = devices.config.enable_rune
          ? training::lesson03_rune_detection(input->image)
          : std::vector<training::Rune>{};
      if (devices.config.display) {
        drawArmors(input->image, armors);
        for (const auto& rune : runes) drawRune(input->image, rune);
      }
      const Eigen::Quaterniond camera_to_world =
          input->has_camera_pose ? input->camera_to_world : input->imu_q;
      const Eigen::Vector3d camera_origin =
          input->has_camera_pose ? input->camera_origin : input->gimbal_origin;
      static double dbg_yaw = 0.0, dbg_pitch = 0.0;
      const auto target = training::lesson04_tracking(
          armors, input->timestamp, input->imu_q, input->gimbal_origin, camera_to_world,
          camera_origin, calibration, tracking);
      if (devices.config.show_diagnostics) {
        static int diag = 0;
        if (++diag % 30 == 0) {
          const double process_ms =
              std::chrono::duration<double>(training::Clock::now() - input->timestamp).count() * 1000.0;
          const double tf_ms = input->tf_sync_ms;
          std::fprintf(stderr,
                       "[diag] frame=%llu dt1=%.2fms tf_dt=%.2fms interp=%d "
                       "yawpitch=(%.1f,%.1f) pos=(%.3f,%.3f,%.3f) vel=(%.3f,%.3f,%.3f)\n",
                       static_cast<unsigned long long>(input->image_id), process_ms, tf_ms,
                       static_cast<int>(input->tf_interpolated),
                       [&] {
                         const Eigen::Matrix3d r = input->imu_q.toRotationMatrix();
                         return std::atan2(r(1, 0), r(0, 0)) * 180.0 / training::kPi;
                       }(),
                       [&] {
                         const Eigen::Matrix3d r = input->imu_q.toRotationMatrix();
                         return std::asin(std::clamp(-r(2, 0), -1.0, 1.0)) * 180.0 / training::kPi;
                       }(),
                       target ? target->position.x() : 0.0,
                       target ? target->position.y() : 0.0,
                       target ? target->position.z() : 0.0,
                       target ? target->velocity.x() : 0.0,
                       target ? target->velocity.y() : 0.0,
                       target ? target->velocity.z() : 0.0);

        }
      }
      // 丢目标时慢速旋转搜索：云台世界 yaw 从底盘姿态 × 云台局部姿态得到，
      // 所以底盘转过之后这里仍然是对的方向。找到目标就自动切回正常瞄准。
      static double search_yaw = 0.0;
      static bool has_search_yaw = false;
      if (devices.config.pose_only) {
        command = training::Command{};  // 独立位姿测量：只观察，不发任何云台/开火指令
      } else if (target) {
        command = training::lesson05_aiming(*target, input->timestamp, input->bullet_speed);
        if (command.control && devices.config.allow_fire && tracking.settle_frames == 0) {
          command.shoot = true;
        }
        if (command.control) has_search_yaw = false;
      } else {
        if (!has_search_yaw) {
          const Eigen::Matrix3d rotation = input->imu_q.toRotationMatrix();
          search_yaw = std::atan2(rotation(1, 0), rotation(0, 0));
          has_search_yaw = true;
        }
        search_yaw += 0.035;  // ≈1.75 rad/s，约 3.6 s 转一圈
        command.control = true;
        command.shoot = false;
        command.yaw = search_yaw;
        command.pitch = 25.0 * M_PI / 180.0;  // 相机水平，方便搜索
      }

      if (devices.config.show_diagnostics) {
        static int cmd_diag = 0;
        if (++cmd_diag % 30 == 0) {
          std::fprintf(stderr, "[cmd] ctrl=%d yaw=%.1f pitch=%.1f shoot=%d\n",
                       static_cast<int>(command.control), command.yaw * 180.0 / training::kPi,
                       command.pitch * 180.0 / training::kPi,
                       static_cast<int>(command.shoot));
        }
      }
      dbg_yaw = command.yaw;
      // 第四讲可视化：PnP 位姿 vs 仿真真值；预测 250ms 后的位置并统计误差。
      static std::deque<std::pair<double, Eigen::Vector3d>> predictions;
      static double last_pred_error = 0.0;
      static bool has_pred_error = false;
      static double last_pose_error = 0.0;
      static bool has_pose_error = false;
      const double now_sec =
          std::chrono::duration<double>(input->timestamp.time_since_epoch()).count();
      constexpr double kPredictAhead = 0.25;  // 250 ms

      if (target) {
        // 真值：/tf 中离 PnP 结果最近的装甲板中心（云台系）。
        // 位姿误差只在近似静止时显示：靶车在动时真值关联和框抖动都会放大误差。
        if (target->velocity.norm() < 0.2) {
          const Eigen::Vector3d in_gimbal = input->imu_q.inverse() * target->position;
          if (const auto gt = training::io::nearestArmorInGimbal(in_gimbal)) {
            last_pose_error = (in_gimbal - *gt).norm();
            has_pose_error = true;
          }
        } else {
          has_pose_error = false;
        }

        const Eigen::Vector3d predicted = target->position + target->velocity * kPredictAhead +
                                          0.5 * target->acceleration * kPredictAhead * kPredictAhead;
        predictions.emplace_back(now_sec, predicted);

        // 画 250ms 后的预测框（青色，调试开关控制）。predicted 是相对云台原点的
        // 向量，投回图像时先加回原点再走 projectWorld。
        if (devices.config.show_prediction && !armors.empty()) {
          if (const auto predicted_px = projectWorldCamera(
                  predicted + input->gimbal_origin, camera_to_world, camera_origin, calibration)) {
            const cv::Rect2f box = cv::boundingRect(armors.front().corners);
            const double current_z = target->position.norm();
            const double predicted_z = predicted.norm();
            const float scale = predicted_z > 1e-3
                ? static_cast<float>(current_z / predicted_z) : 1.0F;
            const cv::Size2f size(box.width * scale, box.height * scale);
            const cv::Rect2f rect(predicted_px->x - size.width * 0.5F,
                                  predicted_px->y - size.height * 0.5F,
                                  size.width, size.height);
            cv::rectangle(input->image, rect, cv::Scalar(255, 200, 0), 2, cv::LINE_AA);
            cv::putText(input->image, "PRED 250ms",
                        cv::Point(static_cast<int>(rect.x), static_cast<int>(rect.y) - 6),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 200, 0), 1, cv::LINE_AA);
          }
        }

        // 250ms 后：预测位置与该时刻实际观测到的目标位置比较，误差显示在左下角。
        while (devices.config.show_prediction && !predictions.empty() &&
               now_sec - predictions.front().first >= kPredictAhead) {
          const Eigen::Vector3d predicted_at = predictions.front().second;
          predictions.pop_front();
          last_pred_error = (predicted_at - target->position).norm();
          has_pred_error = true;
        }
      }

      if (devices.config.display) {
        char line[128];
        if (target) {
          std::snprintf(line, sizeof(line), "PnP dist: %.2f m", target->position.norm());
        } else {
          std::snprintf(line, sizeof(line), "PnP dist: --");
        }
        cv::putText(input->image, line, cv::Point(24, 106), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        if (has_pose_error) {
          std::snprintf(line, sizeof(line), "static pose err vs GT: %.1f cm", last_pose_error * 100.0);
          cv::putText(input->image, line, cv::Point(24, 140), cv::FONT_HERSHEY_SIMPLEX,
                      0.7, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        }
        if (devices.config.show_prediction && has_pred_error) {
          std::snprintf(line, sizeof(line), "pred error @250ms: %.1f cm", last_pred_error * 100.0);
          cv::putText(input->image, line, cv::Point(24, input->image.rows - 24),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        }
      }
    }
    if (interrupted) break;
    devices.write(command);

    if (input && devices.config.display) {
      if (recording) recorder.write(input->image);
      cv::imshow("RoboMaster training", input->image);
      if (!display_configured) {
        // 固定 1:1 显示并挪到左上角，录制时直接截 (0,0)-(1440,1080)。
        cv::moveWindow("RoboMaster training", 0, 0);
        cv::setWindowProperty("RoboMaster training", cv::WND_PROP_TOPMOST, 1);
        display_configured = true;
      }
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q') break;
    }
  }

  devices.close();
}
