# DetectionAI 双目视觉集成开发计划

## 概述

基于当前项目已完成的全部功能（YOLO 检测、NMS、Letterbox、子线程推理、SORT 追踪、越线计数、轨迹可视化、数据导出等），规划双目立体视觉集成方案。目标是在现有架构上分 5 个阶段实现：双目采集 → 相机标定 → 深度计算与测距 → 深度增强追踪 → 硬件扩展。

### 涉及文件总览

| 类型 | 文件 | 职责 |
|------|------|------|
| 新增 | `stereosource.h/cpp` | 双目摄像头硬件抽象 |
| 新增 | `stereotypes.h/cpp` | 标定数据、畸变校正、立体校正 |
| 新增 | `stereomatcher.h/cpp` | SGBM 视差计算、距离估算 |
| 新增 | `calibrationdialog.h/cpp` | 棋盘格标定向导对话框 |
| 新增 | `stereoettingsdialog.h/cpp` | SGBM 参数调节 + 硬件配置 |
| 修改 | `DetectionAI.pro` | 新增源文件和可选硬件依赖 |
| 修改 | `inferencethread.h/cpp` | 双目模式分支、视差计算、距离标注 |
| 修改 | `mainwindow.h/cpp` | 工具栏按钮、深度 Dock、标定入口、设置持久化 |
| 修改 | `yolodetector.h` | Detection 结构体增加 distance 字段 |
| 修改 | `tracker.h/cpp` | Track 增加 distanceHistory 和 avgDistance |
| 修改 | `lang.cpp` | ~80 条中英文字符串 |

---

## 架构设计

### 当前数据流

```
cv::VideoCapture cap_ --> currentFrame_ --> YOLODetector.detect() --> detections
  --> Tracker.update() --> tracks --> 绘制 --> QImage --> frameReady 信号 --> videoLabel_
```

### 双目数据流

在 InferenceThread 内部添加双目分支，与单目路径互斥（`if (stereoMode_)`）。深度计算与检测共享同一线程，保证帧同步。

```
StereoSource (capL_, capR_)
  --> leftFrame_, rightFrame_
  --> StereoRectifier.rectify() (标定后)
  --> StereoMatcher.computeDisparity() --> disparityMap_
  --> YOLODetector.detect(leftRect) --> detections (含 distance)
  --> Tracker.update(detections) --> tracks (含距离历史)
  --> 绘制检测框 + 距离标注 + 可选深度叠加
  --> QImage --> frameReady 信号
  --> depthMapReady 信号 --> 深度 Dock
```

### 类依赖关系

```
MainWindow
  +-- InferenceThread
  |     +-- YOLODetector (已有)
  |     +-- Tracker (已有)
  |     +-- StereoSource (新增) -- 管理双摄像头
  |     +-- StereoMatcher (新增) -- SGBM + 测距
  |     +-- StereoRectifier (新增) -- 校正
  +-- CalibrationDialog (新增) -- 标定向导
  +-- StereoSettingsDialog (新增) -- 参数配置
```

### 关键设计决策

**深度计算在 InferenceThread 内执行**，不另开线程。原因：
1. 深度图必须与检测帧同步，跨线程同步复杂且易出错
2. SGBM 在 640x480 分辨率下 CPU 耗时约 5-15ms，不会显著影响帧率
3. 与现有 `cap_` / `detector_` / `tracker_` 的线程归属一致，无需额外锁

**stereoMode_ 原子标志**控制双目/单目切换。单目路径完全不变，所有新代码在 `if (stereoMode_)` 分支内。

---

## 第一阶段：双摄像头采集与显示

### 目标

实现 StereoSource 硬件抽象，支持两个 USB 摄像头同时采集。InferenceThread 添加双目模式分支，检测在左目帧上运行，右目帧暂不使用（为第二阶段校正做准备）。UI 添加双目模式切换按钮。

### 新增类：StereoSource

**文件：`stereosource.h`**

```cpp
enum class StereoHardware {
    SingleMono,     // 单目回退（默认）
    DualUSB,        // 双独立 USB 摄像头
    DualRTSP,       // 双 RTSP 流
    RealSense,      // Intel RealSense
    ZED             // Stereolabs ZED
};

struct StereoSourceConfig {
    StereoHardware hardware = StereoHardware::SingleMono;
    int leftCameraIndex = 0;
    int rightCameraIndex = 1;
    std::string leftRTSPUrl;
    std::string rightRTSPUrl;
    int targetWidth = 640;
    int targetHeight = 480;
};

class StereoSource {
public:
    StereoSource();
    ~StereoSource();

    bool open(const StereoSourceConfig& config);
    void close();
    bool isOpened() const;

    // 抓取同步帧对（阻塞直到两帧均可用或超时）
    bool grab(cv::Mat& left, cv::Mat& right);

    StereoHardware hardware() const;
    StereoSourceConfig config() const;

    // RealSense/ZED 硬件深度（第五阶段）
    bool hasHardwareDepth() const;
    cv::Mat getHardwareDepth() const;

private:
    StereoSourceConfig config_;
    cv::VideoCapture capL_, capR_;

    bool openDualUSB();
    bool openDualRTSP();
};
```

**文件：`stereosource.cpp`**

`grab()` 实现要点：
- 分别从 capL_ 和 capR_ 抓取帧
- 任一为空则重试（最多 3 次）
- USB 模式下使用 640x480 降低延迟

### InferenceThread 修改

**文件：`inferencethread.h`** — 新增成员

```cpp
StereoSource stereoSource_;
std::atomic<bool> stereoMode_{false};

bool openStereo(const StereoSourceConfig& config);
bool isStereoMode() const;
void setStereoMode(bool enabled);

// 信号
void depthMapReady(const QImage& depthViz, float avgDepth);
```

**文件：`inferencethread.cpp`** — run() 循环修改

在 `while (running_)` 循环开头，将帧获取分支为：

```cpp
if (stereoMode_) {
    cv::Mat leftFrame, rightFrame;
    if (!stereoSource_.grab(leftFrame, rightFrame)) {
        emptyCount++;
        // ... 与单目相同的断连检测逻辑
        continue;
    }
    currentFrame_ = leftFrame;  // 检测运行在左目帧
    rightFrame_ = rightFrame;   // 保存右目帧（后续阶段使用）
} else {
    cap_ >> currentFrame_;
    // ... 原有单目逻辑完全不变
}
```

后续的检测、追踪、绘制逻辑共用（检测在 currentFrame_ 上运行）。

### MainWindow UI 修改

**工具栏新增按钮：**

- `stereoBtn_`（可勾选）— 切换双目/单目模式，快捷键 `B`
- `stereoSettingsBtn_` — 打开双目设置对话框（本阶段为简化版，仅硬件选择）

**双目模式下的行为：**
- 摄像头下拉框隐藏，替换为标签显示 "STEREO: Dual USB"
- 视频画面左上角叠加半透明 "STEREO" 标识
- 检测和追踪功能正常工作

### Lang 字符串（约 15 条）

```
{"stereo_mode",       "双目模式",              "Stereo Mode"},
{"stereo_on",         "关闭双目",              "Stereo On"},
{"stereo_off",        "双目模式",              "Stereo"},
{"tip_stereo",        "双目立体视觉 (B)",       "Stereo vision (B)"},
{"stereo_settings",   "双目设置",              "Stereo Settings"},
{"stereo_hw",         "硬件类型",              "Hardware Type"},
{"hw_dual_usb",       "双USB摄像头",            "Dual USB Cameras"},
{"hw_dual_rtsp",      "双RTSP流",              "Dual RTSP Streams"},
{"hw_realsense",      "Intel RealSense",       "Intel RealSense"},
{"hw_zed",            "Stereolabs ZED",        "Stereolabs ZED"},
{"stereo_open_fail",  "无法打开双目设备",        "Cannot open stereo device"},
{"stereo_left",       "左目",                  "Left"},
{"stereo_right",      "右目",                  "Right"},
{"stereo_url_prompt", "输入左右 RTSP 地址:",    "Enter left/right RTSP URLs:"},
{"stereo_left_url",   "左 RTSP:",              "Left RTSP:"},
{"stereo_right_url",  "右 RTSP:",              "Right RTSP:"},
```

### 设置持久化

新增保存项：

```cpp
settings.setValue("stereoMode", thread_.isStereoMode());
settings.setValue("stereoHardware", (int)config.hardware);
settings.setValue("stereoLeftCam", config.leftCameraIndex);
settings.setValue("stereoRightCam", config.rightCameraIndex);
settings.setValue("stereoLeftRTSP", QString::fromStdString(config.leftRTSPUrl));
settings.setValue("stereoRightRTSP", QString::fromStdString(config.rightRTSPUrl));
```

### 验收标准

- 切换到双目模式后，两个 USB 摄像头同时采集
- 检测在左目帧上正常运行，检测结果与单目模式一致
- 切回单目模式，行为与修改前完全相同（无回归）
- 状态栏显示 "STEREO" 标识

---

## 第二阶段：相机标定

### 目标

实现完整的棋盘格标定工具（内参 + 立体标定），支持加载外部标定文件。创建 StereoRectifier 类执行畸变校正和立体校正。

### 新增类：StereoRectifier

**文件：`stereotypes.h`**

```cpp
struct StereoCalibration {
    cv::Mat cameraMatrixL, distCoeffsL;  // 左相机内参 + 畸变
    cv::Mat cameraMatrixR, distCoeffsR;  // 右相机内参 + 畸变
    cv::Mat R, T;                        // 旋转和平移
    cv::Mat R1, R2, P1, P2, Q;          // 校正变换（stereoRectify 输出）
    cv::Rect validRoiL, validRoiR;       // 校正后有效区域
    bool valid = false;
    double reprojectionError = -1.0;
};

class StereoRectifier {
public:
    StereoRectifier();

    bool loadCalibration(const std::string& path);
    bool saveCalibration(const std::string& path) const;
    void setCalibration(const StereoCalibration& cal);
    bool isCalibrated() const;
    const StereoCalibration& calibration() const;

    // 校正立体帧对（使用预计算 remap，约 1ms）
    void rectify(const cv::Mat& leftRaw, const cv::Mat& rightRaw,
                 cv::Mat& leftRect, cv::Mat& rightRect);

    // 根据标定数据 + 图像尺寸初始化 remap 映射表
    void initRectifyMaps(int imageWidth, int imageHeight);

private:
    StereoCalibration cal_;
    cv::Mat mapL1_, mapL2_;  // 左相机 remap
    cv::Mat mapR1_, mapR2_;  // 右相机 remap
    int lastWidth_ = 0, lastHeight_ = 0;
};
```

**文件：`stereotypes.cpp`** — 核心方法

- `loadCalibration()`: 使用 `cv::FileStorage` 读取 YAML/XML，支持三种格式：
  - 本工具生成的格式
  - ROS camera_info 格式
  - 通用格式（K_left, D_left, K_right, D_right, R, T）
- `rectify()`: 使用 `cv::remap()` 执行预计算映射，每帧约 1ms
- `saveCalibration()`: 写入 YAML 格式

### 新增类：CalibrationDialog

**文件：`calibrationdialog.h`**

```cpp
class CalibrationDialog : public QDialog {
    Q_OBJECT
public:
    explicit CalibrationDialog(StereoSource* source, QWidget* parent = nullptr);
    StereoCalibration result() const;

private slots:
    void onStartCapture();
    void onGrabFrame();
    void onCalibrate();
    void onSaveCalibration();
    void onLoadExternal();

private:
    StereoSource* source_;

    QStackedWidget* stack_;
    // Page 0: 设置页（棋盘尺寸、方格边长）
    QSpinBox* boardWidthSpin_;    // 默认 9
    QSpinBox* boardHeightSpin_;   // 默认 6
    QDoubleSpinBox* squareSizeSpin_;  // 默认 25.0mm
    // Page 1: 采集页（实时预览 + 采集按钮 + 进度）
    QLabel* leftPreview_;
    QLabel* rightPreview_;
    QPushButton* grabBtn_;
    QLabel* captureCountLabel_;
    int capturedCount_ = 0;
    static const int MIN_CAPTURE_FRAMES = 15;
    // Page 2: 结果页（重投影误差 + 保存按钮）
    QLabel* errorLabel_;
    QPushButton* saveBtn_;
    QPushButton* loadExternalBtn_;

    std::vector<std::vector<cv::Point3f>> objectPoints_;
    std::vector<std::vector<cv::Point2f>> imagePointsL_, imagePointsR_;
    cv::Size imageSize_;
    StereoCalibration result_;
    QTimer* previewTimer_;

    void updatePreview();
    bool detectCorners(const cv::Mat& left, const cv::Mat& right);
};
```

**标定流程（三页向导）：**

**第 1 页 — 设置：**
- 输入棋盘格内角数（列 x 行，默认 9x6）
- 输入方格边长（mm，默认 25.0）
- 按钮：取消、下一步

**第 2 页 — 采集：**
- QTimer 驱动实时预览（左右目并排显示）
- 用户点击"采集"或按空格抓取一帧
- `cv::findChessboardCorners()` 检测角点，成功时绿色叠加
- `cv::cornerSubPix()` 亚像素精化
- 进度标签："已采集: 12/25"
- 至少 15 帧后启用"标定"按钮
- 按钮：采集、自动采集、返回、标定

**第 3 页 — 结果：**
- 调用 `cv::calibrateCamera()` 计算各相机内参初值
- 调用 `cv::stereoCalibrate()` 联合标定
- 调用 `cv::stereoRectify()` 计算校正变换
- 显示重投影误差（RMS）：
  - < 1.0 像素：绿色 "优秀"
  - 1.0–2.0 像素：黄色 "可接受"
  - \> 2.0 像素：红色 "较差（建议重新采集）"
- 按钮：保存标定文件、加载外部文件、重新采集、关闭

**外部文件加载：**
- "加载外部标定文件"按钮打开文件对话框
- 支持格式：`.yml` `.yaml` `.xml`
- 加载后自动初始化 rectify maps

### InferenceThread 集成

**新增成员：**

```cpp
StereoRectifier rectifier_;

void setStereoCalibration(const StereoCalibration& cal);
```

**run() 循环中双目分支更新：**

```cpp
if (stereoMode_) {
    cv::Mat leftFrame, rightFrame;
    if (!stereoSource_.grab(leftFrame, rightFrame)) { /* 错误处理 */ }

    if (rectifier_.isCalibrated()) {
        rectifier_.rectify(leftFrame, rightFrame, currentFrame_, rightFrame_);
    } else {
        currentFrame_ = leftFrame;
        rightFrame_ = rightFrame;
    }

    // 检测在校正后的左目帧上运行
    auto dets = detector_.detect(currentFrame_);
    // ... 后续追踪/绘制逻辑不变
}
```

### MainWindow UI 修改

**工具栏新增：**
- `calibrateBtn_` — 打开标定向导（仅双目模式可见）

**快捷键：**
- `Shift+B` — 打开标定对话框

### Lang 字符串（约 25 条）

```
{"calibration",            "标定",                     "Calibration"},
{"calib_start",            "开始标定",                  "Start Calibration"},
{"calib_setup",            "标定设置",                  "Calibration Setup"},
{"calib_board_cols",       "棋盘格内角列数:",           "Board inner cols:"},
{"calib_board_rows",       "棋盘格内角行数:",           "Board inner rows:"},
{"calib_square_size",      "方格边长(mm):",             "Square size (mm):"},
{"calib_capture",          "采集",                     "Capture"},
{"calib_auto_capture",     "自动采集",                  "Auto Capture"},
{"calib_captured",         "已采集: %1/%2",             "Captured: %1/%2"},
{"calib_min_frames",       "至少需要 %1 帧",            "Need at least %1 frames"},
{"calib_calibrating",      "标定计算中...",              "Calibrating..."},
{"calib_done",             "标定完成",                  "Calibration Done"},
{"calib_error",            "重投影误差: %1 像素",        "Reprojection error: %1 px"},
{"calib_quality_good",     "优秀",                     "Excellent"},
{"calib_quality_ok",       "可接受",                    "Acceptable"},
{"calib_quality_poor",     "较差（建议重新采集）",        "Poor (recapture recommended)"},
{"calib_save",             "保存标定文件",               "Save Calibration"},
{"calib_load_external",    "加载外部标定文件",            "Load External Calibration"},
{"calib_file_filter",      "标定文件 (*.yml *.yaml *.xml);;所有文件 (*)",
 "Calibration Files (*.yml *.yaml *.xml);;All Files (*)"},
{"calib_saved",            "标定已保存: ",              "Calibration saved: "},
{"calib_loaded",           "标定已加载: ",              "Calibration loaded: "},
{"calib_load_fail",        "无法加载标定文件。",         "Cannot load calibration file."},
{"calib_no_corners",       "未检测到棋盘角点，请调整棋盘位置。",
 "No chessboard corners detected. Adjust board position."},
{"calib_recapture",        "重新采集",                  "Recapture"},
{"tip_calibrate",          "双目标定 (Shift+B)",         "Stereo calibration (Shift+B)"},
```

### 验收标准

- 使用打印棋盘格完成完整标定流程
- 重投影误差 < 1.0 像素
- 标定数据可保存为 YAML 文件并可重新加载
- 加载外部标定文件（ROS/OpenCV 格式）正常工作
- 校正后的图像极线对齐（同一水平线上）

---

## 第三阶段：深度计算与测距

### 目标

创建 StereoMatcher 类，使用 SGBM 计算视差图，结合 Q 矩阵计算实际距离。Detection 结构体增加 distance 字段，检测框标签显示距离信息。

### 新增类：StereoMatcher

**文件：`stereomatcher.h`**

```cpp
struct SGBMParams {
    int blockSize = 5;           // 匹配块大小（奇数, 3-11）
    int minDisparity = 0;
    int numDisparities = 64;     // 必须为 16 的倍数
    int uniquenessRatio = 10;
    int speckleWindowSize = 100;
    int speckleRange = 32;
    int disp12MaxDiff = 1;
    int preFilterCap = 63;
    int mode = cv::StereoSGBM::MODE_SGBM_3WAY;

    // 距离换算参数
    float baselineMeters = 0.06f;     // 双目基线（米）
    float focalLengthPixels = 700.f;  // 焦距（像素，从标定获取）
};

struct DepthResult {
    float distance;      // 估算距离（米），-1 表示未知
    float confidence;    // 0-1，基于 bbox 内有效视差像素比例
    cv::Mat disparityMap;
};

class StereoMatcher {
public:
    StereoMatcher();

    void setParams(const SGBMParams& params);
    SGBMParams params() const;

    // 从校正后的立体帧对计算视差图
    cv::Mat computeDisparity(const cv::Mat& leftRect, const cv::Mat& rightRect);

    // 从视差图估算 bbox 区域的距离
    DepthResult computeDistance(const cv::Mat& disparityMap, const cv::Rect& bbox) const;

    // 视差图转深度图（米）
    cv::Mat disparityToDepth(const cv::Mat& disparity) const;

    // 生成视差图彩色可视化
    cv::Mat disparityColormap(const cv::Mat& disparity) const;

private:
    SGBMParams params_;
    cv::Ptr<cv::StereoSGBM> sgbm_;
    void createSGBM();
};
```

**文件：`stereomatcher.cpp`** — 核心算法

**`computeDisparity()`：**
- 使用 `cv::StereoSGBM` 计算视差
- 输出 16 位有符号视差图（除以 16 得到实际视差值）

**`computeDistance()`：**
- 在 bbox 区域内采样视差值
- 过滤无效点（视差 <= 0）
- 取中位数视差（比均值更鲁棒）
- 距离公式：`Z = (baseline * focalLength) / disparity`
- 置信度 = 有效像素占比

**`disparityColormap()`：**
- `cv::normalize()` + `cv::applyColorMap(JET)` 生成彩色深度图

### Detection 结构体修改

**文件：`yolodetector.h`**

```cpp
struct Detection {
    int classId;
    float confidence;
    cv::Rect bbox;
    float distance = -1.f;  // 新增：距离（米），-1 表示未知
};
```

向后兼容 — 新字段有默认初始化值，现有代码无需修改。

### InferenceThread 集成

**新增成员：**

```cpp
StereoMatcher matcher_;
std::atomic<bool> depthOverlayEnabled_{false};

void setDepthOverlay(bool enabled);
bool depthOverlayEnabled() const;
cv::Mat lastDisparityMap() const;
```

**run() 循环中双目分支更新：**

```cpp
if (stereoMode_ && rectifier_.isCalibrated()) {
    // 计算视差
    cv::Mat disparity = matcher_.computeDisparity(currentFrame_, rightFrame_);

    // 为每个检测框计算距离
    for (auto& det : dets) {
        auto dr = matcher_.computeDistance(disparity, det.bbox);
        det.distance = dr.distance;
    }
}
```

**绘制距离标签：**

在 `drawDetections()` 和 `drawTracks()` 中，当 `det.distance > 0` 时追加距离：

```cpp
if (det.distance > 0)
    label += " " + cv::format("%.1fm", det.distance);
```

**深度叠加：**

```cpp
if (depthOverlayEnabled_) {
    cv::Mat dispColor = matcher_.disparityColormap(disparity);
    cv::addWeighted(currentFrame_, 0.6, dispColor, 0.4, 0, currentFrame_);
}
```

### MainWindow UI 修改

**工具栏新增：**
- `depthOverlayBtn_`（可勾选）— 深度颜色叠加，快捷键 `D`

**新增深度 Dock：**
- `depthDock_` — 深度数据面板，与 statsDock / countingDock 标签页叠放
- 包含：
  - `depthTable_`：表格显示 追踪ID | 类别 | 距离 | 深度置信度
  - `depthMapLabel_`：缩略深度图（5Hz 更新，避免 UI 开销）

**新增 SGBM 参数调节对话框：**

`StereoSettingsDialog` 增加 SGBM 参数页，滑动条实时调节：
- 匹配块大小、最小视差、视差数量、唯一性比率
- 基线距离(m)、焦距(px)
- 修改即时生效（通过 `setParams()` 传递给 StereoMatcher）

### 新增类：StereoSettingsDialog

**文件：`stereoettingsdialog.h`**

```cpp
class StereoSettingsDialog : public QDialog {
    Q_OBJECT
public:
    explicit StereoSettingsDialog(const StereoSourceConfig& sourceConfig,
                                   const SGBMParams& sgbmParams,
                                   QWidget* parent = nullptr);

    StereoSourceConfig sourceConfig() const;
    SGBMParams sgbmParams() const;
    bool depthOverlayEnabled() const;

private:
    // 硬件选择
    QComboBox* hardwareCombo_;
    QSpinBox* leftCamSpin_;
    QSpinBox* rightCamSpin_;
    QLineEdit* leftRTSPEdit_;
    QLineEdit* rightRTSPEdit_;

    // SGBM 参数
    QSpinBox* blockSizeSpin_;
    QSpinBox* minDisparitySpin_;
    QSpinBox* numDisparitiesSpin_;
    QSpinBox* uniquenessSpin_;
    QSpinBox* speckleWindowSpin_;
    QSpinBox* speckleRangeSpin_;
    QDoubleSpinBox* baselineSpin_;
    QDoubleSpinBox* focalSpin_;

    // 显示选项
    QCheckBox* depthOverlayCheck_;
};
```

### Lang 字符串（约 20 条）

```
{"depth_overlay",      "深度叠加",               "Depth Overlay"},
{"depth_overlay_on",   "关闭深度叠加",            "Hide Depth"},
{"depth_overlay_off",  "深度叠加",               "Depth Overlay"},
{"depth_dock",         "深度数据",               "Depth Data"},
{"depth_map",          "深度图",                 "Depth Map"},
{"distance",           "距离",                   "Distance"},
{"distance_m",         "%1m",                   "%1m"},
{"distance_unknown",   "未知",                   "Unknown"},
{"sgbm_params",        "SGBM 参数",              "SGBM Parameters"},
{"sgbm_block_size",    "匹配块大小",              "Block Size"},
{"sgbm_min_disp",      "最小视差",               "Min Disparity"},
{"sgbm_num_disp",      "视差数量",               "Num Disparities"},
{"sgbm_uniqueness",    "唯一性比率",              "Uniqueness Ratio"},
{"sgbm_speckle_win",   "斑点窗口",               "Speckle Window"},
{"sgbm_speckle_range", "斑点范围",               "Speckle Range"},
{"sgbm_baseline",      "基线距离(m):",            "Baseline (m):"},
{"sgbm_focal",         "焦距(px):",              "Focal Length (px):"},
{"tip_depth_overlay",  "深度颜色叠加 (D)",         "Depth color overlay (D)"},
{"depth_track_id",     "追踪ID",                 "Track ID"},
{"depth_class",        "类别",                   "Class"},
{"depth_dist",         "距离(m)",                "Dist (m)"},
{"depth_conf",         "深度置信度",              "Depth Conf"},
```

### 设置持久化新增

```cpp
settings.setValue("depthOverlay", thread_.depthOverlayEnabled());
settings.setValue("sgbmBlockSize", params.blockSize);
settings.setValue("sgbmNumDisp", params.numDisparities);
settings.setValue("sgbmBaseline", params.baselineMeters);
settings.setValue("stereoCalibPath", lastCalibrationPath_);
```

### 验收标准

- 双目模式 + 标定后，检测框标签显示距离（如 "#3 car 0.87 2.3m"）
- 距离估算在 1-5m 范围内误差 < 10%
- 深度叠加模式下画面显示彩色深度图
- 深度 Dock 实时显示各追踪目标的距离

---

## 第四阶段：深度增强追踪

### 目标

将深度数据集成到 Tracker 中，追踪目标携带距离历史。新增鸟瞰点云 Dock。导出数据包含距离字段。

### Tracker 修改

**文件：`tracker.h`** — Track 结构体扩展

```cpp
struct Track {
    Detection det;       // 已包含 distance 字段
    int trackId;
    std::vector<cv::Point> trajectory;
    std::vector<float> distanceHistory;  // 新增：每帧距离记录
    float speed = 0.f;
    float angle = 0.f;
    float avgDistance = -1.f;            // 新增：平滑平均距离
};
```

**文件：`tracker.cpp`** — update() 方法更新

在匹配成功的轨迹更新中追加：

```cpp
// 更新距离历史
tracks_[i].distanceHistory.push_back(detections[j].distance);
if (tracks_[i].distanceHistory.size() > MAX_TRAJECTORY_LEN)
    tracks_[i].distanceHistory.erase(tracks_[i].distanceHistory.begin());

// 计算平滑平均距离
float sum = 0; int count = 0;
for (float d : tracks_[i].distanceHistory) {
    if (d > 0) { sum += d; count++; }
}
tracks_[i].avgDistance = count > 0 ? sum / count : -1.f;
```

在构建输出 Track 时携带距离数据：

```cpp
result.push_back({{t.classId, t.confidence, t.lastBbox, -1.f},
                  t.trackId, t.trajectory, t.distanceHistory, t.speed, t.angle, t.avgDistance});
```

### InferenceThread 修改

**TrackRecord 扩展：**

```cpp
struct TrackRecord {
    int trackId;
    int classId;
    int64_t timestampMs;
    int x, y, width, height;
    float speed, angle;
    float distance;    // 新增
};
```

**run() 循环中记录追踪历史时增加 distance：**

```cpp
rec.distance = t.det.distance;
```

### 鸟瞰点云 Dock

**MainWindow 新增：**

```cpp
QDockWidget* pointCloudDock_;
QLabel* pointCloudLabel_;
```

**点云生成逻辑（在 InferenceThread 或单独方法中）：**

```cpp
// 从深度图生成鸟瞰图
cv::Mat depth = matcher_.disparityToDepth(disparity);
cv::Mat birdseye(400, 400, CV_8UC3, cv::Scalar(20, 20, 20));

for (int y = 0; y < depth.rows; y += 4) {
    for (int x = 0; x < depth.cols; x += 4) {
        float d = depth.at<float>(y, x);
        if (d <= 0 || d > 10.f) continue;

        // Q 矩阵重投影到 3D
        // 映射到鸟瞰图像素坐标
        int bx = 200 + (int)(/* X * scale */);
        int by = 400 - (int)(d * 40);  // Z 轴映射

        // 颜色根据深度渐变
        cv::circle(birdseye, cv::Point(bx, by), 1, color, -1);
    }
}

// 在鸟瞰图上绘制检测目标位置
for (const auto& t : tracks) {
    if (t.avgDistance > 0) {
        // 绘制彩色圆圈标注
    }
}
```

更新频率约 5Hz，通过独立信号 `pointCloudReady(QImage)` 传递给 UI。

### 越线计数增强

当双目模式启用时，越线统计表新增"穿越距离"列：

| 类别 | 正向 | 反向 | 合计 | 穿越距离(m) |

CrossingEvent 扩展：

```cpp
struct CrossingEvent {
    int trackId;
    int classId;
    int direction;
    int64_t timestampMs;
    float distance;    // 新增：穿越时的距离
};
```

### 导出增强

CSV 追踪数据新增 `distance` 列：

```
trackId,class,classId,timestamp,x,y,width,height,speed,angle,distance
3,car,2,20260429_143012,120,80,60,40,12.5,45.0,2.31
```

JSON 追踪数据中每个 point 新增 `distance` 字段。

### Lang 字符串（约 10 条）

```
{"point_cloud",        "点云视图",               "Point Cloud"},
{"point_cloud_dock",   "鸟瞰点云",               "Bird's Eye View"},
{"pc_top_view",        "俯视图",                 "Top View"},
{"pc_range_m",         "范围: %1m",              "Range: %1m"},
{"depth_avg_dist",     "平均距离",               "Avg Distance"},
{"depth_at_crossing",  "穿越距离(m)",             "Crossing Dist (m)"},
{"depth_export_col",   "distance",               "distance"},
```

### 验收标准

- 追踪目标显示平滑的距离值（非单帧抖动）
- 鸟瞰点云 Dock 实时显示场景俯视投影
- 越线事件记录穿越时的距离
- 导出的 CSV/JSON 包含完整的距离数据

---

## 第五阶段：硬件扩展

### 目标

添加 Intel RealSense 和 Stereolabs ZED 专用支持（硬件深度），完善双 RTSP 同步机制，新增点云文件导出。

### RealSense 支持

**条件编译方式：**

DetectionAI.pro:
```qmake
# 取消注释以启用 RealSense 支持
# DEFINES += USE_REALSENSE
# INCLUDEPATH += "C:/path/to/realsense/include"
# LIBS += -L"C:/path/to/realsense/lib" -lrealsense2
```

**文件：`stereosource.cpp`**

```cpp
#ifdef USE_REALSENSE
#include <librealsense2/rs.hpp>

bool StereoSource::openRealSense() {
    // 使用 RealSense SDK 管道
    // 配置 RGB + Depth 流
    // hasHardwareDepth_ = true
}
#endif
```

当 `hasHardwareDepth()` 返回 true 时，InferenceThread 跳过 SGBM，直接使用硬件深度：

```cpp
if (stereoSource_.hasHardwareDepth()) {
    cv::Mat hwDepth = stereoSource_.getHardwareDepth();
    // 直接使用硬件深度计算距离，精度更高
} else {
    // 使用 SGBM 软件计算
}
```

### ZED 支持

与 RealSense 类似的条件编译方式：

```cpp
#ifdef USE_ZED
#include <sl/Camera.hpp>

bool StereoSource::openZED() {
    // sl::Camera 初始化
    // 配置 left+right + depth 模式
}
#endif
```

### 双 RTSP 同步优化

两个独立 RTSP 流的帧同步策略：

1. 分别抓取左右帧
2. 比较 `cv::CAP_PROP_POS_MSEC` 时间戳
3. 差异 > 100ms 时，丢弃较早的帧并重新抓取
4. 持续漂移时在状态栏警告

```cpp
bool StereoSource::grabDualRTSP(cv::Mat& left, cv::Mat& right) {
    for (int retry = 0; retry < 3; retry++) {
        capL_ >> left;
        capR_ >> right;
        if (left.empty() || right.empty()) continue;

        double tsL = capL_.get(cv::CAP_PROP_POS_MSEC);
        double tsR = capR_.get(cv::CAP_PROP_POS_MSEC);
        if (std::abs(tsL - tsR) < 100.0) return true;
    }
    return false;
}
```

### 点云导出

在导出对话框中新增"点云"选项，支持 PLY 和 XYZ 格式：

**PLY 格式：**
```
ply
format ascii 1.0
element vertex N
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
1.234 0.567 3.890 128 64 32
...
```

**XYZ 格式：**
```
x y z r g b
1.234 0.567 3.890 128 64 32
...
```

### Lang 字符串（约 10 条）

```
{"export_pointcloud",     "导出点云",                    "Export Point Cloud"},
{"export_pc_filter",      "点云 (*.ply *.xyz)",          "Point Cloud (*.ply *.xyz)"},
{"export_pc_done",        "点云已导出: ",                 "Point cloud exported: "},
{"hw_realsense_depth",    "RealSense 硬件深度",           "RealSense Hardware Depth"},
{"hw_zed_depth",          "ZED 硬件深度",                "ZED Hardware Depth"},
{"stereo_sync",           "同步双目帧...",                "Syncing stereo frames..."},
{"stereo_sync_fail",      "双目帧同步失败",               "Stereo frame sync failed"},
{"stereo_hw_depth",       "硬件深度",                    "Hardware Depth"},
```

### 验收标准

- RealSense D400 系列可正常工作，硬件深度精度 < 2%
- ZED 摄像头可正常工作
- 双 RTSP 流帧同步差异 < 100ms
- 点云可导出为 PLY/XYZ 格式

---

## 依赖与构建

### OpenCV 模块

项目已链接 `opencv_world4130`（包含所有模块），双目功能需要的模块：

| 模块 | 使用的函数 | 阶段 |
|------|-----------|------|
| calib3d | `stereoCalibrate`, `stereoRectify`, `findChessboardCorners`, `cornerSubPix`, `initUndistortRectifyMap`, `reprojectImageTo3D` | 2, 3, 4 |
| imgproc | `StereoSGBM`, `applyColorMap` | 3 |
| core | `FileStorage` (YAML/XML I/O) | 2 |

**无需额外安装 OpenCV 库。**

### 可选硬件库

| 库 | 阶段 | 用途 | 必需？ |
|----|------|------|--------|
| librealsense2 | 5 | RealSense 硬件深度 | 可选，`#ifdef USE_REALSENSE` |
| ZED SDK | 5 | ZED 硬件深度 | 可选，`#ifdef USE_ZED` |

### DetectionAI.pro 修改

```qmake
# 新增源文件
SOURCES += stereosource.cpp stereotypes.cpp stereomatcher.cpp \
           calibrationdialog.cpp stereoettingsdialog.cpp

HEADERS += stereosource.h stereotypes.h stereomatcher.h \
           calibrationdialog.h stereoettingsdialog.h

# 可选硬件支持（取消注释启用）
# DEFINES += USE_REALSENSE
# DEFINES += USE_ZED
```

---

## 风险与缓解

### 风险 1：双 USB 摄像头帧同步

**问题**：两个独立 USB 摄像头帧率和时间戳不一致，抓取的帧可能不是同一时刻。

**缓解**：
- USB 模式使用 640x480 低分辨率减少延迟
- 顺序抓取，差异 < 50ms 可接受
- 状态栏显示帧同步质量指标
- 第五阶段硬件摄像头（RealSense/ZED）自带硬件同步

### 风险 2：SGBM CPU 性能

**问题**：SGBM 在 640x480 + 64 视差下 CPU 耗时 15-40ms，可能将帧率从 30fps 降至 15-20fps。

**缓解**：
- 使用 `MODE_SGBM_3WAY`（比完整 SGBM 快）
- 提供快速模式（降低 blockSize 和 speckleWindowSize）
- 可降采样至 320x240 计算视差再上采样叠加
- RealSense/ZED 使用硬件深度，完全跳过 SGBM

### 风险 3：标定质量

**问题**：标定质量差会导致视差图完全不可用。

**缓解**：
- 强制最少 15 帧采集
- 明确显示重投影误差和品质评级
- 支持加载外部标定文件（MATLAB/ROS 专业工具生成）
- 状态栏显示标定质量指示器

### 风险 4：UI 工具栏溢出

**问题**：新增 4-5 个按钮导致工具栏过于拥挤。

**缓解**：
- 双目相关按钮仅在双目模式下显示
- 可将双目按钮组合为下拉菜单按钮
- 备选方案：菜单栏新增"双目"子菜单

### 风险 5：向后兼容

**问题**：修改 Detection 结构体和 run() 循环可能影响单目模式。

**缓解**：
- `Detection::distance` 有默认初始化值 `-1.f`
- 单目路径完全不变（`else` 分支）
- 所有新 UI 控件仅在双目模式激活
- 每个阶段完成后验证单目模式无回归

---

## 实施优先级与时间估算

| 阶段 | 新增文件 | 修改文件 | 预估工期 | 优先级 |
|------|---------|---------|---------|--------|
| 1. 双摄像头采集 | stereotypes | inferencethread, mainwindow, lang, .pro | 1-2 周 | P0 |
| 2. 相机标定 | stereotypes(校正), calibrationdialog | mainwindow, lang | 2-3 周 | P0 |
| 3. 深度计算与测距 | stereomatcher | inferencethread, mainwindow, lang, yolodetector | 1-2 周 | P0 |
| 4. 深度增强追踪 | (无) | tracker, inferencethread, mainwindow, lang | 1-2 周 | P1 |
| 5. 硬件扩展 | (条件编译) | stereosource, mainwindow, lang, .pro | 2-3 周 | P2 |

**建议实施顺序：** Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

先建立双目采集基础和标定能力（Phase 1-2），再实现核心深度功能（Phase 3），然后增强追踪和可视化（Phase 4），最后添加高端硬件支持（Phase 5）。

**总预估工期：** 7-12 周
