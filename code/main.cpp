#include "lessons.hpp"
#include "io_context.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <csignal>
#include <algorithm>
#include <cstdio>
#include <string>

namespace {
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }

// 把第三讲的检测结果画到显示画面上。检测本身在 lesson03_detection 里完成，
// 这里只负责可视化：角点顺序为左上、右上、右下、左下，与 types.hpp 的约定一致。
void drawArmors(cv::Mat& frame, const std::vector<training::Armor>& armors) {
  for (const auto& armor : armors) {
    // corners 是 Point2f，必须逐个取整，不能把 float 直接按 int 重解释。
    const cv::Point polygon[4] = {
        cv::Point(cvRound(armor.corners[0].x), cvRound(armor.corners[0].y)),
        cv::Point(cvRound(armor.corners[1].x), cvRound(armor.corners[1].y)),
        cv::Point(cvRound(armor.corners[2].x), cvRound(armor.corners[2].y)),
        cv::Point(cvRound(armor.corners[3].x), cvRound(armor.corners[3].y)),
    };

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
      if (devices.config.display) drawArmors(input->image, armors);
      const auto target = training::lesson04_tracking(
          armors, input->timestamp, input->imu_q, calibration, tracking);
      if (target)
        command = training::lesson05_aiming(*target, input->timestamp, input->bullet_speed);
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
