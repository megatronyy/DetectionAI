# DetectionAI

基于 Qt 6 + ONNX Runtime + OpenCV 的实时 YOLO11 目标检测桌面应用。支持摄像头、视频文件和 RTSP 网络流输入，实时显示带有检测框、类别标签和追踪 ID 的画面。

## 主要功能

- **实时目标检测**：使用 YOLO11 (yolo11n.onnx) 模型，支持 COCO 80 类目标识别
- **多输入源**：摄像头（0-4 号设备）、本地视频文件（mp4/avi/mkv/mov/wmv）、RTSP/HTTP 网络视频流
- **GPU 加速**：可选 CUDA 推理（编译时启用 `USE_CUDA` 宏），自动回退 CPU
- **NMS 去重**：内置非极大值抑制算法，消除重叠检测框
- **Letterbox 预处理**：保持原始宽高比缩放，避免图像变形
- **实时参数调节**：置信度阈值（0.01-0.99）和 IoU 阈值（0.10-0.90）滑块实时调整
- **检测类别过滤**：通过类别筛选对话框勾选/取消需要检测的类别，支持全选/全不选
- **热切换模型**：运行中切换不同的 ONNX 模型文件，无需重启应用
- **目标追踪（SORT）**：基于卡尔曼滤波 + 匈牙利算法的实时多目标追踪，为每个目标分配持久 ID
- **截图 / 录制**：一键截图保存带检测框的画面，支持录制为 MP4 视频
- **键盘快捷键**：Space 暂停/继续、S 截图、O 打开视频、N 网络摄像头、M 切换模型、T 追踪开关、F11 全屏、Esc 退出
- **设置持久化**：自动保存窗口大小、阈值、摄像头选择、模型路径、追踪状态、类别过滤等用户偏好（QSettings）
- **状态栏信息**：实时显示 FPS、检测目标数、推理设备（CPU/GPU）
- **类别颜色区分**：不同检测类别使用不同颜色（基于黄金角度 HSV 映射）

## 技术栈

| 组件 | 技术 |
|------|------|
| GUI 框架 | Qt 6 (Widgets) |
| 推理引擎 | ONNX Runtime (C++ API) |
| 视频捕获/图像处理 | OpenCV 4.x |
| 构建系统 | qmake |
| C++ 标准 | C++17 |
| 模型格式 | YOLO11 ONNX (yolo11n.onnx) |

## 项目结构

```
DetectionAI/
├── main.cpp                  # 应用入口
├── mainwindow.h/cpp          # 主窗口 UI、控件交互、设置管理
├── mainwindow.ui             # Qt Designer 界面文件
├── yolodetector.h/cpp        # YOLO 推理封装（加载、预处理、推理、后处理、NMS、类别过滤）
├── inferencethread.h/cpp     # 推理线程（帧循环、绘制检测框/追踪框、录制）
├── tracker.h/cpp             # SORT 目标追踪（卡尔曼滤波 + 匈牙利算法）
├── classfilterdialog.h/cpp   # 类别筛选对话框
├── DetectionAI.pro           # qmake 工程文件
├── app.rc / app.ico          # 应用图标资源
├── yolo11n.onnx              # YOLO11 nano 模型文件
└── docs/
    ├── dev_plan.md           # 一期开发计划（已完成）
    └── dev_plan2.md          # 二期开发计划（已完成）
```

## 架构设计

```
┌──────────────┐    signals     ┌──────────────┐
│InferenceThread│ ──────────► │  MainWindow   │
│  (QThread)   │  frameReady  │   (UI 线程)    │
│              │  inputLost   │              │
├──────────────┤              ├──────────────┤
│ VideoCapture │              │  Toolbar     │
│ VideoWriter  │              │  Sliders     │
│ YOLODetector │              │  QLabel      │
│ Tracker      │              └──────────────┘
└──────────────┘
```

- **InferenceThread**：独立线程运行帧捕获 → YOLO 推理 → SORT 追踪 → 绘制检测框 → 录制，通过信号槽将 QImage 传递给 UI 线程
- **YOLODetector**：封装 ONNX Runtime 推理全流程（letterbox → preprocess → Run → postprocess → NMS），支持缓冲区预分配复用和类别过滤
- **Tracker**：实现 SORT 算法，使用 OpenCV 卡尔曼滤波器预测目标运动，匈牙利算法进行检测-轨迹匹配，为跨帧目标分配持久 ID
- **MainWindow**：纯 UI 层，负责控件交互、设置读写、显示渲染后的画面

## 键盘快捷键

| 快捷键 | 功能 |
|--------|------|
| Space | 暂停 / 继续 |
| S | 截图 |
| O | 打开本地视频 |
| N | 打开网络摄像头（RTSP/HTTP） |
| M | 切换 ONNX 模型 |
| T | 开启 / 关闭目标追踪 |
| F11 | 全屏切换 |
| Esc | 退出全屏或关闭窗口 |

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

确保 `yolo11n.onnx` 位于工作目录。启动后应用自动加载上次使用的模型并打开默认摄像头。

## 待办事项

基于已完成的一期和二期开发计划，以下是可进一步完善的功能：

### 中优先级

- [ ] **录制分辨率自适应**：当前录制分辨率硬编码 640x480，应从实际帧尺寸获取
- [ ] **视频循环播放**：视频播完后提供循环播放选项

### 低优先级

- [ ] **多语言支持**：添加英文/中文界面切换
- [ ] **检测统计面板**：统计各类别出现频次、持续时间等
- [ ] **模型性能基准测试**：自动测量并对比不同模型的推理延迟
- [ ] **导出检测结果**：将检测框坐标、类别、置信度导出为 JSON/CSV
