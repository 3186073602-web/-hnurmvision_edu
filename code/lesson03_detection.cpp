#include "lessons.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

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
constexpr double kMinBarArea = 14.0;      // px^2（远距离灯条只有十几像素）

// 灯条配对的几何约束。
constexpr float kMaxHeightDiff = 0.50F;   // 斜视时两侧灯条高度差会变大，放宽
constexpr float kMaxCenterOffset = 0.42F; // 竖直中心差 / 平均灯条高度
constexpr float kMaxTiltDiff = 20.0F;     // 倾斜角差，单位度
constexpr float kMinGapRatio = 0.28F;     // 灯条净间距 / 平均灯条高度
constexpr float kMaxGapRatio = 4.50F;
constexpr float kMinArmorAspect = 1.60F;  // 装甲板外接框 宽/高
constexpr float kMaxArmorAspect = 5.00F;
constexpr float kLargeArmorAspect = 2.85F; // 超过即视为大装甲板（哨兵）
float g_minConfidence = 0.62F;  // 可由 config 调整（0~1）
bool g_keepBestArmor = false;   // 场上多于一个框时只保留置信度最高的那个
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
  // 远处/侧向的灯条偏暗（V 只有 60~120），单独补一张“暗灯条”掩膜，
  // 否则离远了或背光时整块装甲都检不出来。
  cv::Mat blue_dim;
  cv::inRange(hsv, cv::Scalar(85, 95, 45), cv::Scalar(135, 255, 135), blue_dim);
  cv::bitwise_or(blue_low, blue_high, blue_mask);
  cv::bitwise_or(blue_mask, blue_dim, blue_mask);

  cv::inRange(hsv, cv::Scalar(0, 130, 130), cv::Scalar(12, 255, 255), red_low1);
  cv::inRange(hsv, cv::Scalar(168, 130, 130), cv::Scalar(180, 255, 255), red_high1);
  cv::inRange(hsv, cv::Scalar(0, 70, 190), cv::Scalar(12, 255, 255), red_low2);
  cv::inRange(hsv, cv::Scalar(168, 70, 190), cv::Scalar(180, 255, 255), red_high2);
  cv::Mat red_dim;
  cv::inRange(hsv, cv::Scalar(0, 95, 45), cv::Scalar(12, 255, 135), red_dim);
  cv::bitwise_or(red_low1, red_high1, red_mask);
  cv::bitwise_or(red_mask, red_low2, red_mask);
  cv::bitwise_or(red_mask, red_high2, red_mask);
  cv::bitwise_or(red_mask, red_dim, red_mask);

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

void setMinArmorConfidence(float value) {
  g_minConfidence = std::clamp(value, 0.0F, 0.99F);
}

void setKeepBestArmor(bool value) { g_keepBestArmor = value; }

std::vector<Armor> lesson03_detection(const cv::Mat& image) {
  if (std::getenv("LESSON03_DUMP") != nullptr) {
    static int dump_count = 0;
    if (++dump_count % 60 == 0) {
      cv::imwrite("/tmp/det_dump.png", image);
    }
  }
  if (image.empty() || image.type() != CV_8UC3) return {};

  cv::Mat hsv;
  cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  cv::Mat blue_mask, red_mask;
  buildColorMasks(hsv, blue_mask, red_mask);
  cv::Mat color_mask;
  cv::bitwise_or(blue_mask, red_mask, color_mask);

  std::vector<LightBar> bars = findLightBars(blue_mask, 0);
  if (std::getenv("LESSON03_DEBUG") != nullptr) {
    const size_t blue_count = bars.size();
    std::vector<LightBar> probe_red = findLightBars(red_mask, 1);
    std::fprintf(stderr, "[bars] blue=%zu red=%zu\n", blue_count, probe_red.size());
  }
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
      const bool dbg = std::getenv("LESSON03_DEBUG") != nullptr;
      if (height_diff > kMaxHeightDiff) {
        if (dbg) std::fprintf(stderr, "[rej] height %.2f\n", height_diff);
        continue;
      }

      const float center_offset =
          std::abs(left.top.y + left.height * 0.5F - (right.top.y + right.height * 0.5F)) /
          mean_height;
      if (center_offset > kMaxCenterOffset) {
        if (dbg) std::fprintf(stderr, "[rej] center %.2f\n", center_offset);
        continue;
      }

      if (std::abs(left.tilt - right.tilt) > kMaxTiltDiff) continue;

      // 两根灯条粗细要接近：把支架、栏杆上粗细不同的两根竖条排除掉。
      if (std::abs(left.width - right.width) /
              std::max(left.width, right.width) > 0.5F) {
        continue;
      }

      // 净间距 = 两根灯条内侧边之间的距离。
      const float gap = (right.top.x - right.width * 0.5F) -
                        (left.top.x + left.width * 0.5F);
      const float gap_ratio = gap / mean_height;
      const float provisional_width =
          (right.top.x + right.width * 0.5F) - (left.top.x - left.width * 0.5F);
      const bool small_box = provisional_width < 80.0F;  // 远距离的小目标
      if (small_box) {
        if (gap_ratio < 0.20F || gap_ratio > 6.0F) continue;
      } else if (gap_ratio < kMinGapRatio || gap_ratio > kMaxGapRatio) {
        if (dbg) std::fprintf(stderr, "[rej] gap %.2f\n", gap_ratio);
        continue;
      }

      const auto corners = makeCorners(left, right);
      // 用外接矩形判断大小板：斜着看时三角函数的宽度会被低估，bbox 更稳定。
      const cv::Rect2f box = cv::boundingRect(corners);
      if (box.height <= 1.0F) continue;
      const float aspect = box.width / box.height;
      const float min_aspect = box.width < 80.0F ? 1.35F : kMinArmorAspect;
      if (aspect < min_aspect || aspect > kMaxArmorAspect) {
        if (dbg) std::fprintf(stderr, "[rej] aspect %.2f w=%.0f h=%.0f\n", aspect, box.width, box.height);
        continue;
      }

      // 板面检查：真装甲板中间是“暗底 + 数字”，两条灯条只贴在左右边缘。
      // 如果框的中部还出现灯条颜色（说明这个框把另一块真板包了进去，
      // 典型是整车侧面被当成一块板），直接排除。
      if (!color_mask.empty() && box.width >= 80.0F) {
        const int x0 = std::max(0, static_cast<int>(box.x + box.width * 0.30F));
        const int x1 = std::min(color_mask.cols, static_cast<int>(box.x + box.width * 0.70F));
        const int y0 = std::max(0, static_cast<int>(box.y + box.height * 0.15F));
        const int y1 = std::min(color_mask.rows, static_cast<int>(box.y + box.height * 0.85F));
        if (x1 > x0 && y1 > y0) {
          const cv::Rect middle(x0, y0, x1 - x0, y1 - y0);
          const double colored = cv::countNonZero(color_mask(middle));
          const double fraction = colored / static_cast<double>(middle.area());
          if (fraction > 0.12) {
            if (dbg) std::fprintf(stderr, "[rej] interior %.2f w=%.0f\n", fraction, box.width);
            continue;  // 中部有灯条颜色 → 不是一块板
          }
        }
      }

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
      // 近距离的框大，假框（整车侧面）也大，要求更高的置信度；
      // 远距离的框小、评分天然偏低，用较低的阈值，避免远靶直接丢检测。
      const float required_confidence =
          box.width > 120.0F ? std::max(0.80F, g_minConfidence)
                             : (box.width < 80.0F ? 0.50F : g_minConfidence);
      if (confidence < required_confidence) {
        if (dbg) std::fprintf(stderr, "[rej] conf %.2f < %.2f w=%.0f\n", confidence, required_confidence, box.width);
        continue;
      }

      Armor armor;
      // 角点保持灯条外沿（≈ 真实板 135×55 mm），PnP 依赖这个物理尺寸；
      // 显示时再外扩，见 main.cpp 的 drawArmors。
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
    const cv::Rect2f box = cv::boundingRect(armor.corners);
    for (const auto& other : kept) {
      const cv::Rect2f other_box = cv::boundingRect(other.corners);
      // 同一块板的重复检测往往只重叠 30%~40%，单靠 IoU 抑制不掉；
      // 再补一条“中心很近且尺寸接近”的判据，避免同一块板输出两个框。
      const float dx = (box.x + box.width * 0.5F) - (other_box.x + other_box.width * 0.5F);
      const float dy = (box.y + box.height * 0.5F) - (other_box.y + other_box.height * 0.5F);
      const float center_distance = std::sqrt(dx * dx + dy * dy);
      const float width_ratio = box.width / std::max(1.0F, other_box.width);
      const float height_ratio = box.height / std::max(1.0F, other_box.height);
      const bool near_duplicate =
          center_distance < 0.55F * std::max(box.width, other_box.width) &&
          width_ratio > 0.6F && width_ratio < 1.7F &&
          height_ratio > 0.6F && height_ratio < 1.7F;
      // 包住真板的大框（整车侧面被当成一块板）也要抑制：真板几乎完全落在它里面。
      const cv::Rect2f inter = box & other_box;
      const float inter_area = std::max(0.0F, inter.width) * std::max(0.0F, inter.height);
      const float min_area = std::min(box.area(), other_box.area());
      const bool contained = min_area > 1.0F && inter_area / min_area > 0.6F;
      if (std::getenv("LESSON03_DEBUG") != nullptr) {
        std::fprintf(stderr,
                     "[det] box=(%.0f,%.0f,%.0fx%.0f) conf=%.2f other=(%.0f,%.0f,%.0fx%.0f) "
                     "conf=%.2f iou=%.2f near=%d contained=%.2f\n",
                     box.x, box.y, box.width, box.height, armor.confidence,
                     other_box.x, other_box.y, other_box.width, other_box.height,
                     other.confidence, iou(armor.corners, other.corners),
                     static_cast<int>(near_duplicate),
                     min_area > 1.0F ? inter_area / min_area : 0.0F);
      }
      if (iou(armor.corners, other.corners) > kSuppressIou || near_duplicate || contained) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) kept.push_back(armor);
  }
  // 场上出现两个以上框时，只保留置信度（画面里的 SMALL 0.xx）最高的一个。
  if (g_keepBestArmor && kept.size() > 1) {
    const auto best = std::max_element(
        kept.begin(), kept.end(),
        [](const Armor& a, const Armor& b) { return a.confidence < b.confidence; });
    const Armor best_armor = *best;
    kept.clear();
    kept.push_back(best_armor);
  }
  if (std::getenv("LESSON03_DEBUG") != nullptr) {
    std::fprintf(stderr, "[det-final] n=%zu", kept.size());
    for (const auto& armor : kept) {
      const cv::Rect2f box = cv::boundingRect(armor.corners);
      std::fprintf(stderr, " (%.0f,%.0f,%.0fx%.0f c=%.2f)",
                   box.x, box.y, box.width, box.height, armor.confidence);
    }
    std::fprintf(stderr, "\n");
  }
  return kept;
}
}  // namespace training

namespace training {
namespace {

// 能量机关（能量装置）识别：每个黄色圆形小装置单独出一个小框。
//
// 能量机关的叶轮上有若干个黄色圆形 R 标记。按颜色 + 圆度把它们逐个找出来，
// 每个标记画一个紧贴它的小框（不再用一个大框把整个机关包进去）。
struct RuneMark {
  cv::Rect box;
  float circularity = 0;
  double area = 0;
};

std::vector<RuneMark> findRuneMarks(const cv::Mat& image) {
  cv::Mat hsv;
  cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

  cv::Mat mask;
  cv::inRange(hsv, cv::Scalar(18, 120, 150), cv::Scalar(40, 255, 255), mask);
  const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<RuneMark> marks;
  for (const auto& contour : contours) {
    const double area = cv::contourArea(contour);
    if (area < 80.0 || area > 20000.0) continue;

    const double perimeter = cv::arcLength(contour, true);
    if (perimeter <= 1.0) continue;
    const double circularity = 4.0 * kPi * area / (perimeter * perimeter);
    if (circularity < 0.45) continue;

    const cv::Rect box = cv::boundingRect(contour);
    if (box.height <= 0 || box.width <= 0) continue;
    const float aspect = static_cast<float>(box.width) / static_cast<float>(box.height);
    if (aspect < 0.5F || aspect > 2.0F) continue;

    RuneMark mark;
    mark.box = box;
    mark.circularity = static_cast<float>(circularity);
    mark.area = area;
    marks.push_back(mark);
  }

  // 同一个装置可能被闭运算拆成两个候选，按 IoU 去重。
  std::sort(marks.begin(), marks.end(),
            [](const RuneMark& a, const RuneMark& b) { return a.area > b.area; });
  std::vector<RuneMark> kept;
  for (const auto& mark : marks) {
    bool duplicate = false;
    for (const auto& other : kept) {
      const float inter = static_cast<float>((mark.box & other.box).area());
      const float uni = static_cast<float>(mark.box.area() + other.box.area()) - inter;
      if (uni > 0 && inter / uni > 0.5F) { duplicate = true; break; }
    }
    if (!duplicate) kept.push_back(mark);
  }
  return kept;
}
}  // namespace

std::vector<Rune> lesson03_rune_detection(const cv::Mat& image) {
  std::vector<Rune> runes;
  if (image.empty() || image.type() != CV_8UC3) return runes;

  for (const auto& mark : findRuneMarks(image)) {
    // 紧贴标记的小框，留 15% 余量，避免框线压住标记本身。
    const float pad = std::max(mark.box.width, mark.box.height) * 0.15F;
    const cv::Rect2f box(mark.box.x - pad, mark.box.y - pad,
                         mark.box.width + 2 * pad, mark.box.height + 2 * pad);

    Rune rune;
    rune.corners[0] = {box.x, box.y};
    rune.corners[1] = {box.x + box.width, box.y};
    rune.corners[2] = {box.x + box.width, box.y + box.height};
    rune.corners[3] = {box.x, box.y + box.height};
    rune.confidence = std::min(0.99, 0.55 + 0.45 * static_cast<double>(mark.circularity));
    runes.push_back(rune);
  }
  return runes;
}
}  // namespace training
