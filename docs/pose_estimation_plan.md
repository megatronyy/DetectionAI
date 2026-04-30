# DetectionAI 姿态估计功能开发计划

## 概述

在现有 DetectionAI 项目中集成人体姿态估计功能，基于 YOLO11-pose 模型实现 17 个 COCO 关键点检测和骨骼可视化。功能通过扩展现有模块实现，**不新建源文件**，仅修改 7 个文件。

### 设计原则

- **自动识别**：加载模型后根据输出张量形状自动判断检测/姿态模式，无需手动切换
- **向后兼容**：加载普通检测模型时行为完全不变
- **追踪集成**：复用 SORT 追踪器，跨帧跟踪姿态关键点
- **模块扩展**：在现有类中添加功能，不引入新类

### 涉及文件

| 文件 | 修改内容 |
|------|---------|
| `yolodetector.h` | 新增 Keypoint 结构体、扩展 Detection、ModelType 枚举、骨骼常量、关键点名称 |
| `yolodetector.cpp` | loadModel() 中自动检测模型类型、postprocess() 姿态分支、常量定义 |
| `tracker.h/cpp` | InternalTrack 存储 lastKeypoints、输出 Track 携带关键点 |
| `inferencethread.h/cpp` | drawSkeleton()、drawPoseTracks()、骨骼开关、poseDataUpdated 信号 |
| `mainwindow.h/cpp` | 姿态 Dock、骨骼切换按钮、类别筛选联动、导出增强、设置持久化 |
| `lang.cpp` | 约 15 条中英文字符串 |
| `main.cpp` | 注册 metatype |

---

## YOLO11-pose 输出格式

### 与检测模型的区别

| 属性 | YOLO11 检测 | YOLO11-pose |
|------|-----------|-------------|
| 输出形状 | (1, 84, 8400) | (1, 56, 8400) |
| 通道含义 | 4 bbox + 80 类别分数 | 4 bbox + 1 类别分数 + 51 关键点 |
| 类别数 | 80 (COCO) | 1 (仅 person) |
| 通道数 | 4 + NUM_CLASSES = 84 | 4 + 1 + 17×3 = 56 |

### COCO 17 个关键点

| 索引 | 名称 | 索引 | 名称 |
|------|------|------|------|
| 0 | nose（鼻） | 9 | left_wrist（左腕） |
| 1 | left_eye（左眼） | 10 | right_wrist（右腕） |
| 2 | right_eye（右眼） | 11 | left_hip（左髋） |
| 3 | left_ear（左耳） | 12 | right_hip（右髋） |
| 4 | right_ear（右耳） | 13 | left_knee（左膝） |
| 5 | left_shoulder（左肩） | 14 | right_knee（右膝） |
| 6 | right_shoulder（右肩） | 15 | left_ankle（左踝） |
| 7 | left_elbow（左肘） | 16 | right_ankle（右踝） |
| 8 | right_elbow（右肘） | | |

### 骨骼连接定义（16 条连线）

```
头部:  [0,1] [0,2] [1,3] [2,4]
肩部:  [5,6]
左臂:  [5,7] [7,9]
右臂:  [6,8] [8,10]
躯干:  [5,11] [6,12]
髋部:  [11,12]
左腿:  [11,13] [13,15]
右腿:  [12,14] [14,16]
```

---

## 第一部分：数据结构与模型识别

### 1.1 新增 Keypoint 结构体

**文件：`yolodetector.h`**

```cpp
struct Keypoint {
    cv::Point2f pt;       // 原始帧坐标 (x, y)
    float confidence;     // 关键点置信度 [0, 1]
};
```

### 1.2 扩展 Detection 结构体

**文件：`yolodetector.h`**

```cpp
struct Detection {
    int classId;
    float confidence;
    cv::Rect bbox;
    std::vector<Keypoint> keypoints;  // 检测模型为空，姿态模型为 17 个关键点
};
```

向后兼容说明：
- C++17 标准下，结构体成员有默认构造函数时仍支持聚合初始化
- `std::vector` 默认构造为空，聚合初始化中未提供的尾部成员会被值初始化
- 现有代码中 `{cls, conf, cv::Rect(...)}` 形式的构造不受影响，`keypoints` 自动为空 vector
- 需要验证的现有构造点：
  - `yolodetector.cpp:216` — `dets.push_back({cls, conf, cv::Rect(...)});` ✓
  - `tracker.cpp:340` — `result.push_back({{t.classId, t.confidence, t.lastBbox}, ...});` ✓

### 1.3 ModelType 枚举

**文件：`yolodetector.h`**

```cpp
enum class ModelType { Detection, Pose };
```

### 1.4 YOLODetector 新增成员

**文件：`yolodetector.h`**

```cpp
// 公有方法
ModelType modelType() const;
bool isPoseModel() const;
int numKeypoints() const;  // 检测模型返回 0，姿态模型返回实际关键点数

// 私有成员
ModelType modelType_ = ModelType::Detection;
int numKeypoints_ = 0;  // 从输出通道数动态计算
```

### 1.5 常量声明

**文件：`yolodetector.h`** — 声明（与现有 `CLASS_NAMES` 同模式）

```cpp
static const int NUM_KEYPOINTS = 17;
static const int POSE_CHANNELS = 56;   // 4 bbox + 1 conf + 17×3 keypoints

static const std::vector<std::string> KEYPOINT_NAMES;
static const std::vector<std::pair<int,int>> SKELETON_CONNECTIONS;
```

**文件：`yolodetector.cpp`** — 定义（与 `CLASS_NAMES` 同位置）

```cpp
const std::vector<std::string> YOLODetector::KEYPOINT_NAMES = {
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
};

const std::vector<std::pair<int,int>> YOLODetector::SKELETON_CONNECTIONS = {
    {0,1}, {0,2}, {1,3}, {2,4},       // 头部
    {5,6},                              // 肩部
    {5,7}, {7,9},                       // 左臂
    {6,8}, {8,10},                      // 右臂
    {5,11}, {6,12},                     // 躯干
    {11,12},                            // 髋部
    {11,13}, {13,15},                   // 左腿
    {12,14}, {14,16}                    // 右腿
};
```

> **注意**：使用头文件声明 + .cpp 定义的方式（与现有 `CLASS_NAMES` 一致），避免多编译单元 ODR 冲突。不使用头文件内 `static const std::vector<...> = {...}` 赋值。

### 1.6 模型类型自动检测

**文件：`yolodetector.cpp` — `loadModel()` 方法中**

在现有代码缓存 `outNamesC_` 之后（约第 64 行），添加：

```cpp
// 自动检测模型类型
{
    auto outTypeInfo = session_.GetOutputTypeInfo(0);
    auto& tensorInfo = outTypeInfo.GetTensorTypeAndShapeInfo();
    auto outShape = tensorInfo.GetShape();
    int dim1 = (int)outShape[1], dim2 = (int)outShape[2];
    int channels = (dim1 < dim2) ? dim1 : dim2;

    // 通用姿态模型检测：channels = 4 + 1 + numKp * 3
    // 标准 YOLO11-pose: channels = 56 (4+1+17*3)
    int kpCount = (channels - 5) / 3;
    if (channels > 5 && (channels - 5) % 3 == 0 && kpCount > 0) {
        modelType_ = ModelType::Pose;
        numKeypoints_ = kpCount;
    } else {
        modelType_ = ModelType::Detection;
        numKeypoints_ = 0;
    }
}
```

检测逻辑：读取输出张量形状，取较小维度作为通道数。若通道数满足 `4 + 1 + N×3` 格式（N > 0），则为姿态模型，N 为关键点数量。此方式兼容非标准关键点数量的自定义姿态模型。

### 1.7 loadModel() 异常处理中的状态重置

**文件：`yolodetector.cpp` — `loadModel()` 的 catch 块**

现有代码在 `catch(...)` 中仅设置 `loaded_ = false`。需要同步重置模型类型：

```cpp
catch (...) {
    loaded_ = false;
    modelType_ = ModelType::Detection;  // 新增：重置模型类型
    numKeypoints_ = 0;                   // 新增：重置关键点数
    return false;
}
```

---

## 第二部分：推理后处理修改

### 2.1 postprocess() 姿态分支

**文件：`yolodetector.cpp` — `postprocess()` 方法**

在现有 per-box 循环中（提取 bx, by, bw, bh 之后），将置信度计算分支为：

```cpp
if (modelType_ == ModelType::Pose) {
    // 通道 4 = person 类别分数
    float objConf = transposed ? data[4 * numBoxes + i] : ptr[4];
    if (objConf < confThreshold_) continue;

    // 通道 5..(5+numKeypoints_*3-1) = 关键点 × 3 (x, y, conf)
    std::vector<Keypoint> kps;
    kps.reserve(numKeypoints_);
    for (int k = 0; k < numKeypoints_; k++) {
        float kx, ky, kc;
        if (transposed) {
            kx = data[(5 + k*3)     * numBoxes + i];
            ky = data[(5 + k*3 + 1) * numBoxes + i];
            kc = data[(5 + k*3 + 2) * numBoxes + i];
        } else {
            kx = ptr[5 + k*3];
            ky = ptr[5 + k*3 + 1];
            kc = ptr[5 + k*3 + 2];
        }
        // 关键点坐标从 letterbox 空间转换到原始帧空间
        float origX = (kx - padX_) / scaleX_;
        float origY = (ky - padY_) / scaleY_;
        kps.push_back({cv::Point2f(origX, origY), kc});
    }

    // 姿态模型类别固定为 0 (person)
    dets.push_back({0, objConf, cv::Rect(x1, y1, x2 - x1, y2 - y1), kps});
} else {
    // 原有检测逻辑不变
    float conf = 0;
    int cls = 0;
    for (int j = 4; j < channels; j++) {
        float val = transposed ? data[j * numBoxes + i] : ptr[j];
        if (val > conf) { conf = val; cls = j - 4; }
    }
    if (conf < confThreshold_) continue;

    {
        std::lock_guard<std::mutex> lock(filterMutex_);
        if (!enabledClasses_.empty() && !enabledClasses_.contains(cls))
            continue;
    }

    dets.push_back({cls, conf, cv::Rect(x1, y1, x2 - x1, y2 - y1), {}});
}
```

**关键点坐标还原**：与 bbox 坐标相同，使用 letterbox 的 `padX_`、`padY_`、`scaleX_`、`scaleY_` 参数。

**NMS 不变**：现有的 NMS 按 bbox + classId 过滤，对姿态模型同样适用（所有检测 classId=0，同类别内 NMS 正常工作）。

**类别过滤跳过**：姿态模型仅检测 person（classId=0），跳过 enabledClasses_ 过滤。

**动态关键点数量**：使用 `numKeypoints_` 成员变量（从输出通道数计算得到）而非硬编码 17，兼容自定义姿态模型。

---

## 第三部分：追踪器集成

### 3.1 InternalTrack 扩展

**文件：`tracker.h`**

在 `InternalTrack` 结构体中新增：

```cpp
std::vector<Keypoint> lastKeypoints;  // 最新帧的关键点
```

> `tracker.h` 已 `#include "yolodetector.h"`，无需额外 include。

### 3.2 update() 方法修改

**文件：`tracker.cpp`**

**匹配成功的轨迹更新**（约第 272 行之后，`tracks_[i].confidence = ...` 之后）：

```cpp
tracks_[i].lastKeypoints = detections[j].keypoints;
```

**新建轨迹**（约第 315 行之后，`t.confidence = ...` 之后）：

```cpp
t.lastKeypoints = detections[j].keypoints;
```

**输出 Track 构造**（约第 340 行）：

将聚合初始化改为显式构造，确保 keypoints 传递：

```cpp
Track outTrack;
outTrack.det = {t.classId, t.confidence, t.lastBbox, t.lastKeypoints};
outTrack.trackId = t.trackId;
outTrack.trajectory = t.trajectory;
outTrack.speed = speed;
outTrack.angle = angle;
result.push_back(outTrack);
```

替代原来的：
```cpp
result.push_back({{t.classId, t.confidence, t.lastBbox}, t.trackId, t.trajectory, speed, angle});
```

虽然 C++17 聚合初始化会自动将 keypoints 值初始化为空 vector，但显式构造更清晰且确保关键点正确传递。

### 3.3 无需修改 Track 结构体

`Track` 已包含 `Detection det`，而 `Detection` 现在自带 `keypoints`，关键点自动随 Track 流转。现有轨迹线、越线计数、速度方向等功能均基于 bbox 和 trackId，不受关键点影响。

### 3.4 追踪器重置行为

现有 `MainWindow::onToggleTracking()` 在关闭追踪时调用 `thread_.resetTracker()`，这会清空所有 `tracks_`（包括 `lastKeypoints`）。无需额外处理。

---

## 第四部分：可视化绘制

### 4.1 骨骼颜色方案

按身体区域使用不同颜色（BGR 格式）：

| 区域 | 颜色 | BGR 值 | 包含连线 |
|------|------|--------|---------|
| 头部 | 浅蓝 | (255, 203, 192) | [0,1] [0,2] [1,3] [2,4] |
| 手臂 | 紫色 | (192, 128, 255) | [5,7] [7,9] [6,8] [8,10] |
| 躯干 | 绿色 | (128, 255, 128) | [5,6] [5,11] [6,12] [11,12] |
| 腿部 | 橙色 | (96, 224, 255) | [11,13] [13,15] [12,14] [14,16] |
| 关键点 | 黄色 | (0, 255, 255) | 圆点 |

### 4.2 InferenceThread 新增方法

**文件：`inferencethread.h`**

```cpp
// 私有绘制方法
void drawSkeleton(cv::Mat& frame, const Detection& det);
void drawSkeletons(cv::Mat& frame, const std::vector<Detection>& dets);
void drawPoseTracks(cv::Mat& frame, const std::vector<Track>& tracks);

// 骨骼显示开关 + 关键点置信度阈值
std::atomic<bool> skeletonEnabled_{true};
float keypointConfThreshold_ = 0.5f;  // 关键点绘制置信度阈值

// 公有方法
void setSkeletonEnabled(bool enabled);
bool isSkeletonEnabled() const;
void setKeypointConfThreshold(float t);
float keypointConfThreshold() const;
```

### 4.3 drawSkeleton() 实现

**文件：`inferencethread.cpp`**

```cpp
void InferenceThread::drawSkeleton(cv::Mat& frame, const Detection& det)
{
    if (det.keypoints.empty()) return;

    static const int KP_RADIUS = 4;
    static const int LINE_THICKNESS = 2;
    static const cv::Scalar KP_COLOR(0, 255, 255);  // 黄色关键点

    // 按身体区域定义颜色
    static const cv::Scalar HEAD_COLOR(255, 203, 192);   // 浅蓝
    static const cv::Scalar ARM_COLOR(192, 128, 255);    // 紫色
    static const cv::Scalar TORSO_COLOR(128, 255, 128);  // 绿色
    static const cv::Scalar LEG_COLOR(96, 224, 255);     // 橙色

    // 连线索引到颜色的映射（与 SKELETON_CONNECTIONS 顺序对应）
    static const cv::Scalar connColors[] = {
        HEAD_COLOR, HEAD_COLOR, HEAD_COLOR, HEAD_COLOR,  // 0-3: 头部
        TORSO_COLOR,                                       // 4: 肩部
        ARM_COLOR, ARM_COLOR,                              // 5-6: 左臂
        ARM_COLOR, ARM_COLOR,                              // 7-8: 右臂
        TORSO_COLOR, TORSO_COLOR,                          // 9-10: 躯干
        TORSO_COLOR,                                       // 11: 髋部
        LEG_COLOR, LEG_COLOR,                              // 12-13: 左腿
        LEG_COLOR, LEG_COLOR                               // 14-15: 右腿
    };

    // 绘制骨骼连线
    const auto& conns = YOLODetector::SKELETON_CONNECTIONS;
    for (size_t c = 0; c < conns.size(); c++) {
        int a = conns[c].first, b = conns[c].second;
        // 数组越界保护
        if (a >= (int)det.keypoints.size() || b >= (int)det.keypoints.size())
            continue;
        const auto& ka = det.keypoints[a];
        const auto& kb = det.keypoints[b];
        if (ka.confidence < keypointConfThreshold_ ||
            kb.confidence < keypointConfThreshold_)
            continue;
        cv::line(frame, ka.pt, kb.pt, connColors[c], LINE_THICKNESS, cv::LINE_AA);
    }

    // 绘制关键点圆点
    for (const auto& kp : det.keypoints) {
        if (kp.confidence < keypointConfThreshold_) continue;
        cv::circle(frame, kp.pt, KP_RADIUS, KP_COLOR, -1, cv::LINE_AA);
    }
}

void InferenceThread::drawSkeletons(cv::Mat& frame, const std::vector<Detection>& dets)
{
    for (const auto& d : dets)
        drawSkeleton(frame, d);
}
```

**与初始版本的关键改进**：
- 使用 `SKELETON_CONNECTIONS` 常量 + `connColors[]` 数组，而非内联 struct，更清晰
- **数组越界保护**：`if (a >= det.keypoints.size() || b >= det.keypoints.size())` 防止关键点数量 < 17 时崩溃
- **可调节置信度阈值**：使用成员变量 `keypointConfThreshold_` 而非硬编码常量

### 4.4 drawPoseTracks() 实现

复用现有 `drawLabel()` 的标签绘制逻辑，额外叠加骨骼：

```cpp
void InferenceThread::drawPoseTracks(cv::Mat& frame, const std::vector<Track>& tracks)
{
    for (const auto& t : tracks) {
        // 绘制检测框 + 标签
        std::string label = "#" + std::to_string(t.trackId) + " person " +
                            cv::format("%.2f", t.det.confidence);
        if (speedEnabled_ && t.speed > 0.5f)
            label += " " + cv::format("%.1fpx/f", t.speed);
        drawLabel(frame, t.det.bbox, t.det.classId, label);

        // 叠加骨骼
        if (skeletonEnabled_)
            drawSkeleton(frame, t.det);

        // 速度方向箭头
        if (speedEnabled_ && t.speed > 0.5f) {
            cv::Point center(t.det.bbox.x + t.det.bbox.width / 2,
                             t.det.bbox.y + t.det.bbox.height / 2);
            float rad = t.angle * (float)CV_PI / 180.f;
            int arrowLen = std::min(30, (int)(t.speed * 3 + 10));
            cv::Point endPt(center.x + (int)(arrowLen * std::cos(rad)),
                            center.y + (int)(arrowLen * std::sin(rad)));
            cv::Scalar color = classColor(t.det.classId);
            cv::arrowedLine(frame, center, endPt, color, 2, cv::LINE_8, 0, 0.3);
        }
    }
}
```

### 4.5 run() 循环修改

**文件：`inferencethread.cpp` — `run()` 方法**

将现有的追踪/检测绘制分支修改为：

```cpp
if (trackingEnabled_) {
    tracker_.setCurrentTime(QDateTime::currentMSecsSinceEpoch());
    auto tracks = tracker_.update(dets);

    if (detector_.isPoseModel())
        drawPoseTracks(currentFrame_, tracks);
    else
        drawTracks(currentFrame_, tracks);

    if (trajectoryEnabled_)
        drawTrajectory(currentFrame_, tracks);
    if (tracker_.hasCountingLine())
        drawCountingLine(currentFrame_);

    // ... 追踪历史记录（不变）
} else {
    drawDetections(currentFrame_, dets);
    if (detector_.isPoseModel() && skeletonEnabled_)
        drawSkeletons(currentFrame_, dets);
}
```

### 4.6 poseDataUpdated 信号

**文件：`inferencethread.h`** — 新增信号：

```cpp
signals:
    void poseDataUpdated(const std::vector<Detection>& dets);
```

**文件：`inferencethread.cpp`** — run() 中 FPS 计时块内（约第 304 行）：

```cpp
if (detector_.isPoseModel()) {
    emit poseDataUpdated(dets);
}
```

每秒发射一次，携带当前帧的检测结果（含关键点）。

---

## 第五部分：UI 集成

### 5.1 姿态数据 Dock

**文件：`mainwindow.h`** — 新增成员：

```cpp
QDockWidget* poseDock_;
QTableWidget* poseTable_;
QLabel* posePersonLabel_;  // 显示当前选中的人物信息
QSlider* kpConfSlider_;    // 关键点置信度阈值滑动条
QLabel* kpConfValueLabel_;
```

**文件：`mainwindow.cpp`** — `setupUI()` 中新增（在 countingDock 之后）：

```cpp
// --- 姿态数据 dock ---
poseDock_ = new QDockWidget(Lang::s("pose_title"), this);
auto* poseWidget = new QWidget;
auto* poseLayout = new QVBoxLayout(poseWidget);
poseLayout->setContentsMargins(0, 0, 0, 0);

// 人物信息标签
posePersonLabel_ = new QLabel("");
poseLayout->addWidget(posePersonLabel_);

// 关键点表格
poseTable_ = new QTableWidget(0, 3, poseWidget);
poseTable_->setHorizontalHeaderLabels(
    {Lang::s("pose_keypoint"), Lang::s("pose_position"), Lang::s("pose_confidence")});
poseTable_->horizontalHeader()->setStretchLastSection(true);
poseTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
poseTable_->setSelectionBehavior(QAbstractItemView::SelectRows);

poseLayout->addWidget(poseTable_);

// 关键点置信度阈值滑动条
auto* kpConfLayout = new QHBoxLayout;
kpConfLayout->addWidget(new QLabel(Lang::s("kp_conf_threshold")));
kpConfSlider_ = new QSlider(Qt::Horizontal);
kpConfSlider_->setRange(10, 95);
kpConfSlider_->setValue(50);
kpConfSlider_->setMinimumWidth(100);
kpConfValueLabel_ = new QLabel("0.50");
kpConfValueLabel_->setMinimumWidth(36);
kpConfLayout->addWidget(kpConfSlider_);
kpConfLayout->addWidget(kpConfValueLabel_);
poseLayout->addLayout(kpConfLayout);

connect(kpConfSlider_, &QSlider::valueChanged, this, [this](int value) {
    float threshold = value / 100.f;
    thread_.setKeypointConfThreshold(threshold);
    kpConfValueLabel_->setText(QString::number(threshold, 'f', 2));
});

poseDock_->setWidget(poseWidget);
addDockWidget(Qt::RightDockWidgetArea, poseDock_);
tabifyDockWidget(statsDock_, poseDock_);

// 默认隐藏，加载姿态模型后显示
poseDock_->hide();
```

**姿态数据更新**：显示置信度最高的人的关键点信息。

**文件：`mainwindow.h`** — 新增 slot：

```cpp
void onPoseDataUpdated(const std::vector<Detection>& dets);
```

```cpp
void MainWindow::onPoseDataUpdated(const std::vector<Detection>& dets)
{
    if (!poseDock_->isVisible()) return;

    if (dets.empty()) {
        poseTable_->setRowCount(0);
        posePersonLabel_->setText("");
        return;
    }

    // 取置信度最高的人（NMS 后不一定按置信度排序）
    const auto& det = *std::max_element(dets.begin(), dets.end(),
        [](const Detection& a, const Detection& b) { return a.confidence < b.confidence; });

    posePersonLabel_->setText(
        QString("Person conf: %1  |  Keypoints: %2")
            .arg(det.confidence, 0, 'f', 2)
            .arg(det.keypoints.size()));

    poseTable_->setRowCount((int)det.keypoints.size());
    for (int i = 0; i < (int)det.keypoints.size(); i++) {
        const auto& kp = det.keypoints[i];
        QString name = (i < (int)YOLODetector::KEYPOINT_NAMES.size())
            ? QString::fromStdString(YOLODetector::KEYPOINT_NAMES[i])
            : QString("kp_%1").arg(i);
        poseTable_->setItem(i, 0, new QTableWidgetItem(name));
        poseTable_->setItem(i, 1,
            new QTableWidgetItem(QString("(%1, %2)").arg((int)kp.pt.x).arg((int)kp.pt.y)));
        poseTable_->setItem(i, 2,
            new QTableWidgetItem(QString::number(kp.confidence, 'f', 2)));
    }
}
```

### 5.2 骨骼切换按钮

**文件：`mainwindow.h`** — 新增成员：

```cpp
QPushButton* skeletonBtn_;
```

**文件：`mainwindow.cpp`** — 工具栏中添加（在 `speedBtn_` 之后）：

```cpp
skeletonBtn_ = new QPushButton(Lang::s("skeleton_off"));
skeletonBtn_->setCheckable(true);
skeletonBtn_->setChecked(true);
skeletonBtn_->setToolTip(Lang::s("tip_skeleton"));
toolbar->addWidget(skeletonBtn_);

connect(skeletonBtn_, &QPushButton::toggled, this, &MainWindow::onToggleSkeleton);
```

默认隐藏，加载姿态模型后显示：
```cpp
skeletonBtn_->setVisible(thread_.detector().isPoseModel());
```

**新增 slot 声明（mainwindow.h）：**

```cpp
void onToggleSkeleton(bool checked);
```

```cpp
void MainWindow::onToggleSkeleton(bool checked)
{
    thread_.setSkeletonEnabled(checked);
    skeletonBtn_->setText(checked ? Lang::s("skeleton_on") : Lang::s("skeleton_off"));
}
```

### 5.3 快捷键

**文件：`mainwindow.cpp`** — `keyPressEvent()` 中添加：

```cpp
case Qt::Key_K:
    if (thread_.detector().isPoseModel())
        skeletonBtn_->toggle();
    break;
```

### 5.4 模型加载后的 UI 响应

在构造函数和 `loadModelFile()` 中，模型加载成功后，统一调用以下 UI 更新逻辑（提取为公共方法避免重复）：

**文件：`mainwindow.h`** — 新增私有方法：

```cpp
void updateModelTypeUI();
```

**文件：`mainwindow.cpp`** — 实现：

```cpp
void MainWindow::updateModelTypeUI()
{
    bool isPose = thread_.detector().isPoseModel();

    // 姿态专属 UI 可见性
    poseDock_->setVisible(isPose);
    skeletonBtn_->setVisible(isPose);

    // 姿态模式下禁用类别筛选（仅检测 person，筛选无意义）
    classFilterBtn_->setVisible(!isPose);

    // 状态栏提示
    if (isPose)
        statusBar()->showMessage(Lang::s("pose_model_loaded"), 3000);
    else
        statusBar()->showMessage(Lang::s("detection_model_loaded"), 3000);

    // 设备标签增加模型类型提示
    QString devText = thread_.detector().isGpuEnabled()
        ? Lang::s("device_gpu") : Lang::s("device_cpu");
    if (isPose) devText += " | Pose";
    deviceLabel_->setText(devText);
}
```

在以下位置调用 `updateModelTypeUI()`：
1. 构造函数中，模型加载成功后
2. `loadModelFile()` 中，模型加载成功后
3. `dropEvent()` 中加载 .onnx 文件成功后（走 `loadModelFile()` 已覆盖）

### 5.5 信号连接

**文件：`mainwindow.cpp`** — 构造函数中：

```cpp
connect(&thread_, &InferenceThread::poseDataUpdated,
        this, &MainWindow::onPoseDataUpdated);
```

**文件：`main.cpp`** — 注册 metatype：

```cpp
#include "yolodetector.h"  // 确保 Detection 类型完整

// 在 main() 中，已有的 qRegisterMetaType 之后添加
qRegisterMetaType<std::vector<Detection>>("std::vector<Detection>");
```

### 5.6 关于对话框动态更新

**文件：`mainwindow.cpp`** — `onAbout()` 方法

当前 about 文本硬编码 "COCO 80 类"。在姿态模式下应显示不同信息：

```cpp
void MainWindow::onAbout()
{
    QString cvVer = QString("%1.%2.%3")
        .arg(CV_VERSION_MAJOR).arg(CV_VERSION_MINOR).arg(CV_VERSION_REVISION);

    if (thread_.detector().isPoseModel()) {
        QMessageBox::about(this, Lang::s("about"),
            Lang::s("about_text_pose").arg(qVersion()).arg(cvVer));
    } else {
        QMessageBox::about(this, Lang::s("about"),
            Lang::s("about_text").arg(qVersion()).arg(cvVer));
    }
}
```

---

## 第六部分：设置持久化

### 6.1 loadSettings()

```cpp
bool skeleton = settings.value("skeleton", true).toBool();
skeletonBtn_->setChecked(skeleton);
thread_.setSkeletonEnabled(skeleton);

float kpConf = settings.value("keypointConfThreshold", 0.5).toFloat();
kpConfSlider_->setValue((int)(kpConf * 100));
thread_.setKeypointConfThreshold(kpConf);

if (settings.value("poseDockVisible", false).toBool())
    poseDock_->show();
else
    poseDock_->hide();
```

### 6.2 saveSettings()

```cpp
settings.setValue("skeleton", thread_.isSkeletonEnabled());
settings.setValue("keypointConfThreshold", thread_.keypointConfThreshold());
settings.setValue("poseDockVisible", poseDock_->isVisible());
```

---

## 第七部分：导出增强

### 7.1 检测结果导出（JSON）

在 JSON 导出中，当 `d.keypoints` 非空时追加关键点数据：

```cpp
if (!d.keypoints.empty()) {
    QJsonArray kpArr;
    for (int k = 0; k < (int)d.keypoints.size(); k++) {
        QJsonObject kpObj;
        QString name = (k < (int)YOLODetector::KEYPOINT_NAMES.size())
            ? QString::fromStdString(YOLODetector::KEYPOINT_NAMES[k])
            : QString("kp_%1").arg(k);
        kpObj["name"] = name;
        kpObj["x"] = qRound(d.keypoints[k].pt.x * 10) / 10.0;
        kpObj["y"] = qRound(d.keypoints[k].pt.y * 10) / 10.0;
        kpObj["confidence"] = qRound(d.keypoints[k].confidence * 1000) / 1000.0;
        kpArr.append(kpObj);
    }
    obj["keypoints"] = kpArr;
}
```

JSON 输出示例：

```json
{
  "timestamp": "2026-04-29T14:30:12",
  "detections": [
    {
      "class": "person",
      "classId": 0,
      "confidence": 0.87,
      "bbox": [100, 80, 60, 180],
      "keypoints": [
        {"name": "nose",           "x": 140.5, "y": 60.2,  "confidence": 0.95},
        {"name": "left_eye",       "x": 135.2, "y": 55.8,  "confidence": 0.88},
        {"name": "right_eye",      "x": 145.8, "y": 56.1,  "confidence": 0.91},
        {"name": "left_shoulder",  "x": 120.3, "y": 95.7,  "confidence": 0.82},
        {"name": "right_shoulder", "x": 160.7, "y": 96.2,  "confidence": 0.85},
        ...
      ]
    }
  ]
}
```

### 7.2 检测结果导出（CSV）

修改表头：

```
class,classId,confidence,x,y,width,height,keypoints
```

关键点列格式（分号分隔，每个关键点 `name:x,y,conf`）：

```
person,0,0.870,100,80,60,180,"nose:140.5,60.2,0.950;left_eye:135.2,55.8,0.880;..."
```

### 7.3 追踪数据导出

TrackRecord 新增关键点字段：

```cpp
struct TrackRecord {
    int trackId;
    int classId;
    int64_t timestampMs;
    int x, y, width, height;
    float speed, angle;
    std::vector<float> kpData;  // [x0,y0,c0, x1,y1,c1, ...] 平铺存储
};
```

在 run() 循环记录历史时填充：

```cpp
rec.kpData.reserve(t.det.keypoints.size() * 3);
for (const auto& kp : t.det.keypoints) {
    rec.kpData.push_back(kp.pt.x);
    rec.kpData.push_back(kp.pt.y);
    rec.kpData.push_back(kp.confidence);
}
```

JSON 追踪数据导出示例：

```json
{
  "timestamp": "2026-04-29T14:30:12",
  "tracks": [
    {
      "trackId": 3,
      "class": "person",
      "classId": 0,
      "points": [
        {
          "t": "2026-04-29T14:30:12.500",
          "x": 120, "y": 80, "w": 60, "h": 40,
          "speed": 12.5, "angle": 45.0,
          "keypoints": [
            {"name": "nose", "x": 140.5, "y": 60.2, "conf": 0.95},
            {"name": "left_eye", "x": 135.2, "y": 55.8, "conf": 0.88},
            ...
          ]
        }
      ]
    }
  ],
  "uniqueCounts": {"person": 5}
}
```

---

## 第八部分：refreshUIText() 增强

```cpp
// 骨骼按钮
if (skeletonBtn_) {
    skeletonBtn_->setText(skeletonBtn_->isChecked() ? Lang::s("skeleton_on") : Lang::s("skeleton_off"));
    skeletonBtn_->setToolTip(Lang::s("tip_skeleton"));
}

// 姿态 Dock
poseDock_->setWindowTitle(Lang::s("pose_title"));
poseTable_->setHorizontalHeaderLabels(
    {Lang::s("pose_keypoint"), Lang::s("pose_position"), Lang::s("pose_confidence")});
```

---

## 第九部分：Lang 字符串

```cpp
// 姿态估计
{"pose_title",           "姿态数据",                    "Pose Data"},
{"pose_keypoint",        "关键点",                      "Keypoint"},
{"pose_position",        "位置",                       "Position"},
{"pose_confidence",      "置信度",                      "Confidence"},
{"skeleton_on",          "隐藏骨骼",                    "Hide Skeleton"},
{"skeleton_off",         "显示骨骼",                    "Skeleton"},
{"tip_skeleton",         "显示骨骼连线和关键点 (K)",      "Show skeleton & keypoints (K)"},
{"pose_model_loaded",    "姿态模型已加载",               "Pose model loaded"},
{"detection_model_loaded","检测模型已加载",              "Detection model loaded"},
{"kp_conf_threshold",    "关键点阈值:",                  "KP Threshold:"},
{"about_text_pose",      "<h2>DetectionAI</h2>"
                         "<p>YOLO11 人体姿态估计</p>"
                         "<p>基于 Qt %1 / OpenCV %2 / ONNX Runtime</p>"
                         "<p>COCO 17 关键点 | SORT 多目标追踪</p>",
 "<h2>DetectionAI</h2>"
 "<p>YOLO11 Human Pose Estimation</p>"
 "<p>Powered by Qt %1 / OpenCV %2 / ONNX Runtime</p>"
 "<p>COCO 17 Keypoints | SORT Multi-Object Tracking</p>"},
```

---

## 实施顺序

由于模块间存在依赖关系，按以下顺序实施：

| 步骤 | 文件 | 内容 | 依赖 |
|------|------|------|------|
| 1 | `yolodetector.h` | Keypoint 结构体、Detection 扩展、ModelType 枚举、常量声明 | 无 |
| 2 | `yolodetector.cpp` | 常量定义、loadModel() 检测模型类型、postprocess() 姿态分支 | 步骤 1 |
| 3 | `tracker.h/cpp` | InternalTrack 存储 lastKeypoints、输出携带关键点 | 步骤 1 |
| 4 | `lang.cpp` | 新增中英文字符串（含 about_text_pose） | 无 |
| 5 | `inferencethread.h/cpp` | drawSkeleton()、drawPoseTracks()、run() 姿态分支、信号、置信度阈值 | 步骤 1-3 |
| 6 | `mainwindow.h/cpp` | 姿态 Dock、骨骼按钮、类别筛选联动、关于对话框、导出增强、设置持久化、updateModelTypeUI() | 步骤 4-5 |
| 7 | `main.cpp` | 注册 metatype | 步骤 1 |

---

## 风险与缓解

### 风险 1：检测模型输出通道数恰好匹配姿态格式

**问题**：非标准检测模型的通道数可能满足 `(channels - 5) % 3 == 0`。

**缓解**：
- 标准 YOLO11 检测模型输出 84 通道（4 + 80 类），不满足条件
- 自定义模型若碰巧匹配，可通过文件名启发式（含 "pose" 则为姿态）或增加额外判断
- 当前检测逻辑覆盖标准使用场景，非标准模型需要用户自行适配

### 风险 2：骨骼绘制性能

**问题**：每人物绘制 17 个关键点 + 16 条连线，多人场景可能影响帧率。

**缓解**：
- OpenCV 的 `cv::line()` 和 `cv::circle()` 为高效 CPU 绘制原语，10 人以内影响可忽略
- `skeletonEnabled_` 开关提供性能逃生通道
- 关键点置信度阈值过滤低质量关键点，减少绘制数量
- 实测若影响明显可降低关键点绘制频率（隔帧绘制）

### 风险 3：关键点精度

**问题**：YOLO11n-pose 是轻量模型，关键点精度有限，远距离或遮挡时关键点不准确。

**缓解**：
- 关键点置信度阈值（默认 0.5，可通过 Dock 中滑动条调节）过滤低质量关键点
- 不绘制低置信度的连线和圆点，避免视觉噪声
- 可通过 `kpConfSlider_` 实时调节阈值

### 风险 4：向后兼容

**问题**：修改 Detection 结构体可能破坏现有代码。

**缓解**：
- `keypoints` 使用 `std::vector`（默认构造为空），C++17 下聚合初始化不受影响
- 现有代码中所有 `{cls, conf, rect}` 构造仍有效（已验证 yolodetector.cpp:216 和 tracker.cpp:340）
- 检测模型路径（postprocess 的 else 分支）完全不变
- 加载检测模型时所有姿态 UI 隐藏，类别筛选恢复显示

### 风险 5：静态常量 ODR 冲突

**问题**：`KEYPOINT_NAMES` 和 `SKELETON_CONNECTIONS` 若在头文件中赋值，多编译单元包含时会导致重复定义。

**缓解**：采用与现有 `CLASS_NAMES` 相同的模式 — 头文件声明、.cpp 定义。

---

## 验收标准

1. 加载 `yolo11n-pose.onnx` 自动进入姿态模式，状态栏显示 "姿态模型已加载"
2. 视频画面显示人物关键点（黄色圆点）和骨骼连线（彩色分区域）
3. 骨骼可通过工具栏按钮或快捷键 K 切换显示/隐藏
4. 姿态数据 Dock 显示关键点名称、坐标、置信度，自动选中置信度最高的人物
5. Dock 中关键点置信度滑动条可实时调节骨骼显示灵敏度
6. 姿态模式下类别筛选按钮自动隐藏
7. 追踪模式下，同一人物跨帧保持追踪 ID 和骨骼显示
8. 轨迹线、越线计数、速度方向等现有功能在姿态模式下正常工作
9. 关于对话框在姿态模式下显示 "COCO 17 关键点 | SORT 多目标追踪"
10. 切换回 `yolo11n.onnx` 检测模型，行为与修改前完全一致
11. 导出 JSON/CSV 包含完整关键点数据
12. 设置（骨骼开关、关键点阈值、Dock 可见性）持久保存
13. 拖放 .onnx 姿态模型文件到窗口自动识别并切换
