#include "lessons.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace training {
namespace {

// ---------------------------------------------------------------------------
// 第三讲：装甲板检测（传统视觉 / OpenCV）
//
// 机甲大师的装甲板由“左右两条竖立的彩色灯条 + 中间的数字/图案”组成，
// 因此检测分两步：
//   1. 在 HSV 空间按红/蓝两个色相区间提取灯条候选，用形态学把被反光打断的
//      灯条碎片接回去；
//   2. 枚举同一颜色的灯条对，用“高度接近、竖直中心对齐、倾斜角接近、间距与
//      灯条高度成比例”四条几何约束筛出装甲板，再按宽高比区分大小板。
//
// 输入图像是 1440x1080 的相机画面，坐标系为图像坐标（原点左上、y 向下）。
// ---------------------------------------------------------------------------

// 灯条候选的几何过滤。数值按 1440x1080 相机标定，换相机后需要重调。
constexpr float kMinBarHeight = 16.0F;    // px，太小的灯条不可靠
constexpr float kMinBarAspect = 1.55F;    // 高/宽，灯条是细长条
constexpr float kMaxBarWidth = 90.0F;     // px，太宽的多半是车体反光
constexpr double kMinBarArea = 35.0;      // px^2

// 灯条配对的几何约束。
constexpr float kMaxHeightDiff = 0.40F;   // 两条灯条高度差 / 较高者
constexpr float kMaxCenterOffset = 0.42F; // 竖直中心差 / 平均灯条高度
constexpr float kMaxTiltDiff = 20.0F;     // 倾斜角差，单位度
constexpr float kMinGapRatio = 0.28F;     // 灯条净间距 / 平均灯条高度
constexpr float kMaxGapRatio = 4.50F;
constexpr float kMinArmorAspect = 1.35F;  // 装甲板外接框 宽/高
constexpr float kMaxArmorAspect = 5.00F;
constexpr float kLargeArmorAspect = 2.85F; // 超过即视为大装甲板（哨兵）
constexpr float kMinConfidence = 0.55F;
constexpr float kSuppressIou = 0.45F;

struct LightBar {
  cv::Point2f top;    // 灯条上端短边的中点
  cv::Point2f bottom; // 灯条下端短边的中点
  float height = 0.0F;
  float width = 0.0F;
  float tilt = 0.0F;  // 相对竖直方向的倾角，度，左倾为负
  int color = 0;      // 0 = 蓝，1 = 红
};

// 把 HSV 中属于“灯条”的像素按颜色分到两张掩膜里。灯条是饱和度高、亮度高的
// 彩色发光条；车体、地面是低饱和或低亮度的，会在这一步被排除。
void buildColorMasks(const cv::Mat& hsv, cv::Mat& blue_mask, cv::Mat& red_mask) {
  cv::Mat blue_low, blue_high, red_low1, red_high1, red_low2, red_high2;
  cv::inRange(hsv, cv::Scalar(85, 120, 120), cv::Scalar(135, 255, 255), blue_low);
  cv::inRange(hsv, cv::Scalar(85, 60, 190), cv::Scalar(135, 255, 255), blue_high);
  cv::bitwise_or(blue_low, blue_high, blue_mask);

  cv::inRange(hsv, cv::Scalar(0, 130, 130), cv::Scalar(12, 255, 255), red_low1);
  cv::inRange(hsv, cv::Scalar(168, 130, 130), cv::Scalar(180, 255, 255), red_high1);
  cv::inRange(hsv, cv::Scalar(0, 70, 190), cv::Scalar(12, 255, 255), red_low2);
  cv::inRange(hsv, cv::Scalar(168, 70, 190), cv::Scalar(180, 255, 255), red_high2);
  cv::bitwise_or(red_low1, red_high1, red_mask);
  cv::bitwise_or(red_mask, red_low2, red_mask);
  cv::bitwise_or(red_mask, red_high2, red_mask);

  // 灯条偶尔会被号码/支架切断，闭运算把同一根灯条上的碎片接回去。
  const cv::Mat kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 9));
  cv::morphologyEx(blue_mask, blue_mask, cv::MORPH_CLOSE, kernel);
  cv::morphologyEx(red_mask, red_mask, cv::MORPH_CLOSE, kernel);
}

// 从一张颜色掩膜里提取灯条候选。
std::vector<LightBar> findLightBars(const cv::Mat& mask, int color) {
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<LightBar> bars;
  for (const auto& contour : contours) {
    const double area = cv::contourArea(contour);
    if (area < kMinBarArea) continue;

    const cv::RotatedRect rect = cv::minAreaRect(contour);
    const float long_side = std::max(rect.size.width, rect.size.height);
    const float short_side = std::min(rect.size.width, rect.size.height);
    if (long_side < kMinBarHeight) continue;
    if (short_side <= 0.5F) continue;
    if (long_side / short_side < kMinBarAspect) continue;
    if (short_side > kMaxBarWidth) continue;

    // 用矩形顶点确定上/下端短边中点：按 y 排序，前两个点构成上端边。
    std::array<cv::Point2f, 4> pts{};
    rect.points(pts.data());
    std::sort(pts.begin(), pts.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });

    LightBar bar;
    bar.color = color;
    bar.top = (pts[0] + pts[1]) * 0.5F;
    bar.bottom = (pts[2] + pts[3]) * 0.5F;
    bar.height = static_cast<float>(cv::norm(bar.bottom - bar.top));
    if (bar.height < kMinBarHeight) continue;
    bar.width = static_cast<float>(area) / bar.height;
    bar.tilt = static_cast<float>(
        std::atan2(bar.top.x - bar.bottom.x, bar.bottom.y - bar.top.y) * 180.0 /
        kPi);

    // 只保留竖直方向为主的灯条，斜躺的长条（场地线、车体边缘）直接丢弃。
    if (std::abs(bar.tilt) > 55.0F) continue;
    bars.push_back(bar);
  }
  return bars;
}

// 由一对灯条构造装甲板的四个角点，顺序固定为左上、右上、右下、左下。
std::array<cv::Point2f, 4> makeCorners(const LightBar& left, const LightBar& right) {
  std::array<cv::Point2f, 4> corners{};
  corners[0] = {left.top.x - left.width * 0.5F, left.top.y};
  corners[1] = {right.top.x + right.width * 0.5F, right.top.y};
  corners[2] = {right.bottom.x + right.width * 0.5F, right.bottom.y};
  corners[3] = {left.bottom.x - left.width * 0.5F, left.bottom.y};
  return corners;
}

float quadWidth(const std::array<cv::Point2f, 4>& c) {
  return 0.5F * (static_cast<float>(cv::norm(c[1] - c[0])) +
                 static_cast<float>(cv::norm(c[2] - c[3])));
}

float quadHeight(const std::array<cv::Point2f, 4>& c) {
  return 0.5F * (static_cast<float>(cv::norm(c[3] - c[0])) +
                 static_cast<float>(cv::norm(c[2] - c[1])));
}

// 两框交并比，用于去掉同一块装甲板被重复配对出来的结果。
float iou(const std::array<cv::Point2f, 4>& a, const std::array<cv::Point2f, 4>& b) {
  const cv::Rect2f ra = cv::boundingRect(a);
  const cv::Rect2f rb = cv::boundingRect(b);
  const float inter = (ra & rb).area();
  const float uni = ra.area() + rb.area() - inter;
  return uni > 0.0F ? inter / uni : 0.0F;
}

}  // namespace

std::vector<Armor> lesson03_detection(const cv::Mat& image) {
  if (image.empty() || image.type() != CV_8UC3) return {};

  cv::Mat hsv;
  cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

  cv::Mat blue_mask, red_mask;
  buildColorMasks(hsv, blue_mask, red_mask);

  std::vector<LightBar> bars = findLightBars(blue_mask, 0);
  std::vector<LightBar> red_bars = findLightBars(red_mask, 1);
  bars.insert(bars.end(), red_bars.begin(), red_bars.end());

  // 按 x 排序后，同一块装甲板的左灯条一定排在右灯条前面。
  std::sort(bars.begin(), bars.end(), [](const LightBar& a, const LightBar& b) {
    return a.top.x < b.top.x;
  });

  std::vector<Armor> armors;
  for (std::size_t i = 0; i < bars.size(); ++i) {
    for (std::size_t j = i + 1; j < bars.size(); ++j) {
      const LightBar& left = bars[i];
      const LightBar& right = bars[j];
      if (left.color != right.color) continue;

      const float mean_height = 0.5F * (left.height + right.height);
      const float height_diff =
          std::abs(left.height - right.height) / std::max(left.height, right.height);
      if (height_diff > kMaxHeightDiff) continue;

      const float center_offset =
          std::abs(left.top.y + left.height * 0.5F - (right.top.y + right.height * 0.5F)) /
          mean_height;
      if (center_offset > kMaxCenterOffset) continue;

      if (std::abs(left.tilt - right.tilt) > kMaxTiltDiff) continue;

      // 净间距 = 两根灯条内侧边之间的距离。
      const float gap = (right.top.x - right.width * 0.5F) -
                        (left.top.x + left.width * 0.5F);
      const float gap_ratio = gap / mean_height;
      if (gap_ratio < kMinGapRatio || gap_ratio > kMaxGapRatio) continue;

      const auto corners = makeCorners(left, right);
      // 用外接矩形判断大小板：斜着看时三角函数的宽度会被低估，bbox 更稳定。
      const cv::Rect2f box = cv::boundingRect(corners);
      if (box.height <= 1.0F) continue;
      const float aspect = box.width / box.height;
      if (aspect < kMinArmorAspect || aspect > kMaxArmorAspect) continue;

      // 置信度由“高度一致 / 倾斜一致 / 间距合理”三项加权，作为后续跟踪的
      // 观测权重，也用于视频里的可视化。
      const float size_score = 1.0F - height_diff;
      const float angle_score =
          1.0F - std::abs(left.tilt - right.tilt) / kMaxTiltDiff;
      const float ideal_gap = 1.65F;  // 小装甲板“灯条间距/灯条高度”的经验值
      const float gap_score =
          std::exp(-0.5F * std::pow((gap_ratio - ideal_gap) / 0.9F, 2.0F));
      const float confidence =
          0.45F * size_score + 0.30F * angle_score + 0.25F * gap_score;
      if (confidence < kMinConfidence) continue;

      Armor armor;
      armor.corners = corners;
      armor.large_armor = aspect > kLargeArmorAspect;
      armor.class_id = -1;  // 本讲不做数字识别，无法读取编号
      armor.confidence = std::min(0.99F, std::max(0.0F, confidence));
      armors.push_back(armor);
    }
  }

  // 同一块装甲板可能有多组灯条组合得到相近的框，按置信度做一次 NMS。
  std::sort(armors.begin(), armors.end(),
            [](const Armor& a, const Armor& b) { return a.confidence > b.confidence; });
  std::vector<Armor> kept;
  for (const auto& armor : armors) {
    bool duplicate = false;
    for (const auto& other : kept) {
      if (iou(armor.corners, other.corners) > kSuppressIou) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) kept.push_back(armor);
  }
  return kept;
}
}  // namespace training
