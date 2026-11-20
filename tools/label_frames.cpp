// 用第三讲的检测结果给仿真帧生成 YOLO 标注（单类装甲板，0=small，1=large）。
// 仅用于课程自建数据集：仿真帧没有人工标注，先用传统检测器产生伪标签，
// 再人工抽查修正。用法：
//   ./label_frames <图片目录> <标签目录> <可视化目录>
#include "lessons.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: label_frames <images> <labels> <viz>\n";
    return 2;
  }
  const fs::path images_dir = argv[1];
  const fs::path labels_dir = argv[2];
  const fs::path viz_dir = argv[3];
  fs::create_directories(labels_dir);
  fs::create_directories(viz_dir);

  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(images_dir)) {
    const auto ext = entry.path().extension().string();
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") files.push_back(entry.path());
  }
  std::sort(files.begin(), files.end());

  int kept = 0;
  int skipped = 0;
  for (const auto& file : files) {
    const cv::Mat image = cv::imread(file.string(), cv::IMREAD_COLOR);
    if (image.empty()) continue;
    const auto armors = training::lesson03_detection(image);
    if (armors.empty()) {
      ++skipped;
      continue;
    }

    cv::Mat viz = image.clone();
    std::ofstream label(labels_dir / (file.stem().string() + ".txt"));
    for (const auto& armor : armors) {
      const cv::Rect2f box = cv::boundingRect(armor.corners);
      // 过滤贴边、过小的框，避免把远处噪声当成训练目标。
      if (box.width < 18 || box.height < 24) continue;
      const float cx = (box.x + box.width * 0.5F) / image.cols;
      const float cy = (box.y + box.height * 0.5F) / image.rows;
      const float w = box.width / image.cols;
      const float h = box.height / image.rows;
      const int cls = armor.large_armor ? 1 : 0;
      label << cls << ' ' << cx << ' ' << cy << ' ' << w << ' ' << h << '\n';

      const cv::Scalar color = armor.large_armor ? cv::Scalar(0, 165, 255)
                                                 : cv::Scalar(0, 255, 0);
      cv::rectangle(viz, box, color, 2);
      char text[64];
      std::snprintf(text, sizeof(text), "%s %.2f", armor.large_armor ? "L" : "S",
                    armor.confidence);
      cv::putText(viz, text, cv::Point(box.x, std::max(12.0F, box.y - 6.0F)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
    label.close();
    cv::imwrite((viz_dir / file.filename()).string(), viz);
    ++kept;
    if (kept % 50 == 0) std::cout << "labeled " << kept << std::endl;
  }
  std::cout << "done: labeled " << kept << ", skipped " << skipped << std::endl;
  return 0;
}
