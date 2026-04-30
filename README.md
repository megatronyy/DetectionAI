# DetectionAI

基于 Qt 6 + ONNX Runtime + OpenCV 的实时 YOLO11 目标检测桌面应用。支持摄像头自动检测、视频文件、拖放打开和 RTSP 网络流输入，实时显示带有检测框、类别标签和追踪 ID 的画面。支持中英文界面切换。

## 主要功能

### 检测与推理
- **实时目标检测**：使用 YOLO11 模型，支持 COCO 80 类目标识别
- **人体姿态估计**：自动识别 YOLO11-pose 模型（输出通道数检测），支持 COCO 17 关键点检测与骨骼连线可视化
- **GPU 加速**：可选 CUDA 推理（编译时启用 `USE_CUDA` 宏），自动回退 CPU
- **NMS 去重**：内置非极大值抑制算法，消除重叠检测框
- **Letterbox 预处理**：保持原始宽高比缩放，避免图像变形
- **实时参数调节**：置信度阈值（0.01-0.99）和 IoU 阈值（0.10-0.90）滑块实时调整
- **检测类别过滤**：通过类别筛选对话框勾选/取消需要检测的类别，支持全选/全不选
- **热切换模型**：运行中切换不同的 ONNX 模型文件（检测/姿态），自动适配模型类型
- **最近模型列表**：快速访问最近使用的 5 个模型
- **目标追踪（SORT）**：基于卡尔曼滤波 + 匈牙利算法的实时多目标追踪，为每个目标分配持久 ID

### 姿态估计
- **自动模型检测**：根据 ONNX 输出张量形状自动判断是检测模型还是姿态模型（channels-5 能否被 3 整除）
- **骨骼连线渲染**：按身体区域（头部/上肢/下肢）分色绘制骨骼连线和关键点圆圈
- **关键点置信度过滤**：可调节关键点置信度阈值，低于阈值的关键点和连线不显示
- **姿态追踪**：追踪模式下为每个姿态目标分配持久 ID 并绘制轨迹
- **姿态数据面板**：右侧可停靠面板，实时显示选中目标的各关键点名称、坐标、置信度

### 双目立体视觉
- **双目模式**：支持双 USB 摄像头、双 RTSP 流输入，运行时一键切换
- **棋盘格标定**：三步标定向导（设置 → 采集 → 结果），自动检测棋盘角点，计算双目标定参数和重投影误差
- **外部标定加载**：支持加载已有的 YAML/XML 标定文件
- **SGBM 深度估计**：基于 Semi-Global Block Matching 计算视差图和目标距离（Z = baseline × focal / disparity）
- **深度颜色叠加**：将深度图以 JET 伪彩色叠加到检测画面上
- **深度数据面板**：显示各检测目标的实时距离和深度置信度
- **鸟瞰点云视图**：将深度图以俯视角度可视化
- **SGBM 参数调节**：独立设置面板，可调节匹配块大小、视差范围、基线距离、焦距等参数
- **深度增强追踪**：追踪目标携带距离历史和平均距离信息，越线事件记录穿越时距离
- **硬件扩展**：预留 Intel RealSense 和 Stereolabs ZED 接口（条件编译）

### 追踪增强
- **轨迹线可视化**：为每个追踪目标绘制彩色运动轨迹，渐变粗细显示移动历史（上限 200 点）
- **唯一目标计数**：基于 trackId 去重统计每个类别的独立目标数量，区别于检测次数累计
- **速度与方向估算**：从卡尔曼滤波器状态提取目标速度和运动方向，标签显示速度值并在目标中心绘制方向箭头
- **越线计数**：在画面上交互式画一条虚拟计数线，基于叉积符号翻转检测目标穿越，区分正/反向越线并实时统计各类别越线次数
- **追踪数据导出**：支持将完整追踪历史（轨迹、速度、方向、越线事件）导出为 JSON 或 CSV 格式，JSON 按目标 ID 分组

### 输入源
- **摄像头自动检测**：启动时自动枚举可用摄像头，无需手动猜测设备号
- **本地视频文件**：支持 mp4/avi/mkv/mov/wmv 格式
- **RTSP/HTTP 网络流**：输入 URL 即可拉取网络视频流
- **双目输入**：双 USB 摄像头或双 RTSP 流同步采集
- **视频循环播放**：可切换的循环播放模式，视频播完后自动从头播放
- **拖放打开**：直接拖放视频文件或 ONNX 模型到窗口即可打开

### 输出与统计
- **截图 / 录制**：一键截图保存带检测框的画面，支持录制为 MP4 视频（自动适配源分辨率）
- **导出检测结果**：支持 JSON 和 CSV 两种格式导出检测框坐标、类别、置信度、关键点数据；开启追踪时可选择导出追踪数据
- **检测统计面板**：右侧可停靠面板，实时累计显示各类别检测次数和唯一目标数，支持清零统计和重置计数
- **越线统计面板**：独立可停靠面板，按类别显示正向/反向/合计越线次数，支持清除计数线
- **深度数据面板**：双目模式下显示各追踪目标的实时距离（米）和深度置信度
- **鸟瞰点云面板**：可视化深度图的俯视投影
- **推理耗时显示**：状态栏实时显示每帧推理耗时（毫秒）

### 界面与交互
- **中英文切换**：一键切换中文/英文界面，所有 UI 文字实时更新
- **工具栏 Tooltips**：悬停按钮显示功能描述和对应快捷键
- **窗口位置记忆**：自动保存并恢复窗口位置和大小
- **交互式画线**：点击工具栏按钮进入画线模式，在视频画面上依次点击两个点定义计数线，支持输入标签和 Escape 取消
- **键盘快捷键**：Space 暂停/继续、S 截图、O 打开视频、N 网络摄像头、M 切换模型、T 追踪开关、Shift+T 轨迹线、Shift+S 速度方向、K 骨骼显示、B 双目模式、Shift+B 标定、D 深度叠加、C 画计数线、L 循环播放、E 导出检测、F11 全屏、Esc 取消画线/退出全屏/关闭
- **关于对话框**：显示应用版本及 Qt/OpenCV 版本信息，自动适配检测/姿态模型类型
- **设置持久化**：自动保存所有用户偏好（阈值、语言、追踪状态、轨迹线、速度方向、骨骼、循环播放、类别过滤、双目配置、SGBM 参数、标定路径、面板可见性等）
- **状态栏信息**：实时显示 FPS、检测目标数、推理耗时、推理设备（CPU/GPU | Pose | Stereo）
- **类别颜色区分**：不同检测类别使用不同颜色（基于黄金角度 HSV 映射）

## 技术栈

| 组件 | 技术 |
|------|------|
| GUI 框架 | Qt 6 (Widgets) |
| 推理引擎 | ONNX Runtime (C++ API) |
| 视频捕获/图像处理 | OpenCV 4.x |
| 构建系统 | qmake |
| C++ 标准 | C++17 |
| 模型格式 | YOLO11 ONNX (检测/姿态) |

## 项目结构

```
DetectionAI/
├── main.cpp                    # 应用入口，注册跨线程元类型
├── mainwindow.h/cpp            # 主窗口 UI、控件交互、设置管理、拖放、画线、双目 UI
├── yolodetector.h/cpp          # YOLO 推理封装（检测/姿态自动识别、预处理、推理、后处理、NMS）
├── inferencethread.h/cpp       # 推理线程（帧循环、双目分支、深度叠加、绘制、录制）
├── tracker.h/cpp               # SORT 目标追踪（卡尔曼滤波 + 匈牙利算法 + 速度 + 越线 + 距离）
├── stereosource.h/cpp          # 双目硬件抽象（双 USB / 双 RTSP / RealSense / ZED 接口）
├── stereotypes.h/cpp           # 双目标定数据结构 + StereoRectifier（remap 畸变校正）
├── stereomatcher.h/cpp         # SGBM 视差计算 + 目标距离估算 + 深度着色
├── calibrationdialog.h/cpp     # 棋盘格标定向导（三步：设置 → 采集 → 结果）
├── stereoettingsdialog.h/cpp   # 双目设置对话框（硬件选择 + SGBM 参数）
├── classfilterdialog.h/cpp     # 类别筛选对话框
├── lang.h/cpp                  # 中英文翻译映射系统
├── DetectionAI.pro             # qmake 工程文件
├── app.rc / app.ico            # 应用图标资源
├── yolo11n.onnx                # YOLO11 nano 检测模型
└── docs/
    ├── dev_plan.md             # 一期开发计划（已完成）
    ├── dev_plan2.md            # 二期开发计划（已完成）
    ├── dev_plan3.md            # 三期开发计划（已完成）
    ├── pose_estimation_plan.md # 姿态估计计划（已完成）
    └── binocular_plan.md       # 双目视觉计划（已完成）
```

## 架构设计

```
┌──────────────────┐    signals     ┌──────────────┐
│ InferenceThread   │ ──────────► │  MainWindow   │
│   (QThread)       │  frameReady  │   (UI 线程)    │
│                   │  inputLost   │              │
│                   │  tracking    ├──────────────┤
│                   │  StatsUpdated│  Toolbar     │
│                   │  crossing    │  Sliders     │
│                   │  poseData    │  QLabel      │
│                   │  depthMap    │  Stats Dock  │
├───────────────────┤              │  Crossing    │
│ VideoCapture      │              │  Pose Dock   │
│ VideoWriter       │              │  Depth Dock  │
│ YOLODetector      │              │  Point Cloud │
│ Tracker           │              └──────────────┘
│ StereoSource      │
│ StereoRectifier   │
│ StereoMatcher     │
└───────────────────┘
```

- **InferenceThread**：独立线程运行帧捕获 →（双目分支：立体采集 → 畸变校正 → SGBM 视差 → 深度叠加）→ YOLO 推理 → SORT 追踪（含越线检测）→ 绘制 → 录制，通过信号槽将 QImage、推理耗时、类别统计、唯一计数、越线统计、姿态数据、深度图传递给 UI 线程
- **YOLODetector**：封装 ONNX Runtime 推理全流程，自动检测模型类型（检测 vs 姿态），姿态模型动态计算关键点数量，支持关键点后处理和骨骼绘制
- **Tracker**：实现 SORT 算法，卡尔曼滤波预测 + 匈牙利匹配；扩展支持轨迹历史、唯一 ID 计数、速度/方向提取、越线检测、距离历史和平均距离
- **StereoSource**：硬件抽象层，支持双 USB 摄像头和双 RTSP 流同步采集，预留 RealSense/ZED 硬件深度接口
- **StereoRectifier**：基于 cv::remap 的双目畸变校正，支持加载/保存标定文件（YAML/XML）
- **StereoMatcher**：封装 cv::StereoSGBM 视差计算，支持中值距离估算和 JET 伪彩色深度图
- **CalibrationDialog**：三步棋盘格标定向导，自动角点检测 + stereoCalibrate + stereoRectify
- **Lang**：轻量翻译系统，字符串键值映射中英文，支持运行时切换语言并刷新所有 UI 文字
- **MainWindow**：纯 UI 层，负责控件交互、设置读写、面板更新（检测统计/越线/姿态/深度/点云）、检测/追踪导出、语言切换、拖放、画线交互

## 键盘快捷键

| 快捷键 | 功能 |
|--------|------|
| Space | 暂停 / 继续 |
| S | 截图 |
| Shift+S | 开启 / 关闭速度方向显示 |
| O | 打开本地视频 |
| N | 打开网络摄像头（RTSP/HTTP） |
| M | 切换 ONNX 模型 |
| T | 开启 / 关闭目标追踪 |
| Shift+T | 开启 / 关闭轨迹线 |
| K | 开启 / 关闭骨骼显示（姿态模型） |
| B | 开启 / 关闭双目模式 |
| Shift+B | 打开双目标定 |
| D | 开启 / 关闭深度叠加（双目模式） |
| C | 画计数线 |
| L | 开启 / 关闭循环播放 |
| E | 导出检测结果或追踪数据（JSON/CSV） |
| F11 | 全屏切换 |
| Esc | 取消画线 / 退出全屏 / 关闭窗口 |

## 构建与运行

### 前置依赖

1. **Qt 6** (Widgets 模块)
2. **OpenCV 4.x**（本项目使用 opencv_world 4.13.0）
3. **ONNX Runtime**（CPU 版本即可，GPU 版需额外 CUDA Provider）
4. **MSVC 2022**（Windows 平台）

### 配置路径

编辑 `DetectionAI.pro` 中的路径变量以匹配本地安装：

```qmake
OPENCV_INC = C:/path/to/opencv/build/include
OPENCV_LIB = C:/path/to/opencv/build/x64/vc16/lib
ORT_INC    = C:/path/to/onnxruntime/include
ORT_LIB    = C:/path/to/onnxruntime/lib
```

启用 GPU 推理：取消注释 `DEFINES += USE_CUDA`。

### 构建

使用 Qt Creator 打开 `DetectionAI.pro`，或命令行：

```bash
qmake DetectionAI.pro
nmake debug   # Windows MSVC
```

### 运行

确保 `yolo11n.onnx` 位于工作目录。启动后应用自动加载上次使用的模型、枚举可用摄像头，用户偏好（窗口位置、阈值、语言、追踪状态、双目配置、SGBM 参数等）自动恢复。
