# DetectionAI 开发计划

基于当前项目状态，按优先级排列待完善功能。

---

## 第一阶段：核心检测质量（必做）

### 1. 添加 NMS（非极大值抑制）
- **现状**：同一物体会产生多个重叠检测框，影响可用性
- **方案**：在 `postprocess` 中实现标准 NMS 算法
  - 按置信度降序排列候选框
  - 逐个选取最高分框，删除与它 IoU > 阈值（建议 0.45）的所有剩余框
- **涉及文件**：`mainwindow.cpp`、`mainwindow.h`（新增 `float IOU_THRESHOLD` 常量和 `float computeIOU()` 方法）

### 2. 优化预处理性能
- **现状**：`preprocess` 中双层 for 循环逐像素拷贝 HWC→CHW，效率低
- **方案**：用 OpenCV 的 `split()` + `reshape()` 或直接内存拷贝替代逐像素访问
- **涉及文件**：`mainwindow.cpp`

### 3. Letterbox 预处理（保持宽高比）
- **现状**：直接 resize 到 640×640 会拉伸图像，导致检测结果变形
- **方案**：按原始宽高比缩放，不足部分用灰色填充（letterbox），并记录缩放偏移量用于后处理坐标还原
- **涉及文件**：`mainwindow.cpp`、`mainwindow.h`（新增 letterbox 相关参数）

---

## 第二阶段：UI 与交互体验

### 4. 状态栏显示 FPS 和检测数量
- **现状**：无法直观了解推理性能和检测情况
- **方案**：在窗口底部状态栏显示实时 FPS、检测目标数
- **涉及文件**：`mainwindow.cpp`、`mainwindow.h`

### 5. 摄像头选择与视频文件输入
- **现状**：硬编码打开设备 0，无其他输入方式
- **方案**：
  - 下拉框选择摄像头设备索引
  - 支持打开本地视频文件（mp4/avi）
- **涉及文件**：`mainwindow.cpp`、`mainwindow.h`、`mainwindow.ui`

### 6. 开始/暂停/截图控制
- **现状**：启动即开始检测，无法暂停或保存画面
- **方案**：
  - 添加工具栏按钮：开始、暂停、截图保存
  - 截图保存为带检测框的图片文件
- **涉及文件**：`mainwindow.cpp`、`mainwindow.h`、`mainwindow.ui`

### 7. 置信度阈值实时调节
- **现状**：阈值硬编码为 0.25，修改需重新编译
- **方案**：添加滑动条控件，实时调节置信度阈值
- **涉及文件**：`mainwindow.cpp`、`mainwindow.h`、`mainwindow.ui`

---

## 第三阶段：工程化完善

### 8. GPU 推理支持（CUDA）
- **现状**：仅支持 CPU 推理
- **方案**：检测是否有 CUDA provider，有则自动启用 GPU 推理，无则回退 CPU
- **涉及文件**：`mainwindow.cpp`、`DetectionAI.pro`（条件链接 onnxruntime providers 库）

### 9. 错误处理与用户提示
- **现状**：模型加载失败、摄像头打开失败仅输出 qDebug，用户无感知
- **方案**：用 QMessageBox 弹窗提示关键错误；模型文件不存在时弹出文件选择对话框
- **涉及文件**：`mainwindow.cpp`

### 10. 代码结构重构
- **现状**：所有逻辑集中在 MainWindow，不利于维护和扩展
- **方案**：将 YOLO 推理逻辑独立为 `YOLODetector` 类（封装 init/preprocess/infer/postprocess），MainWindow 只负责 UI 和调度
- **涉及文件**：新建 `yolodetector.h/cpp`，修改 `mainwindow.h/cpp`、`DetectionAI.pro`

---

## 建议实施顺序

1 → 3 → 2 → 9 → 4 → 7 → 6 → 5 → 10 → 8

先提升检测质量（NMS + letterbox），再改善用户体验（错误提示 + FPS + 阈值调节 + 控制按钮），最后做工程化重构和 GPU 支持。
