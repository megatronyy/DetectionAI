# 目标追踪功能增强开发计划

## 概述

当前项目已实现 SORT 多目标追踪（Kalman 滤波 + 匈牙利算法），支持跨帧为检测目标分配唯一 ID 并可视化显示。本计划在此基础上扩展 5 项功能。

### 涉及文件

| 文件 | 职责 |
|------|------|
| `tracker.h/cpp` | SORT 核心算法，InternalTrack 结构体 |
| `inferencethread.h/cpp` | 推理线程，绘制逻辑，run() 主循环 |
| `mainwindow.h/cpp` | UI 控件，工具栏，Dock 窗口，设置持久化 |
| `lang.cpp` | 中英文字符串表 |
| `classfilterdialog.h/cpp` | 对话框参考模式 |

---

## 功能一：轨迹线可视化

### 目标
在视频画面上绘制每个追踪目标的运动轨迹路径，直观展示目标的移动历史。

### 实现方案

#### 1. Tracker 层 — 存储轨迹历史

**文件：`tracker.h`**

```cpp
struct InternalTrack {
    // ...existing fields...
    std::vector<cv::Point> trajectory;  // 新增：轨迹点历史
};
```

**文件：`tracker.cpp`**

在 `update()` 方法中，匹配成功的轨迹追加当前帧的中心点：

```cpp
// Step 4: Update matched tracks — 在已有更新逻辑后追加
tracks_[i].trajectory.push_back(
    cv::Point(detections[j].bbox.x + detections[j].bbox.width / 2,
              detections[j].bbox.y + detections[j].bbox.height / 2));

// 限制轨迹长度，避免内存持续增长
if (tracks_[i].trajectory.size() > 200)
    tracks_[i].trajectory.erase(tracks_[i].trajectory.begin());
```

新增轨迹输出结构体字段：

```cpp
struct Track {
    Detection det;
    int trackId;
    std::vector<cv::Point> trajectory;  // 新增
};
```

在 `update()` 返回结果时携带轨迹数据。

#### 2. InferenceThread 层 — 绘制轨迹线

**文件：`inferencethread.h`**

```cpp
std::atomic<bool> trajectoryEnabled_{false};
void setTrajectoryEnabled(bool enabled);
bool isTrajectoryEnabled() const;
```

**文件：`inferencethread.cpp`**

新增 `drawTrajectory()` 方法：

```cpp
void InferenceThread::drawTrajectory(cv::Mat& frame, const std::vector<Track>& tracks)
{
    for (const auto& t : tracks) {
        if (t.trajectory.size() < 2) continue;
        cv::Scalar color = classColor(t.det.classId);
        // 渐变透明效果：越早的点越细
        for (size_t i = 1; i < t.trajectory.size(); i++) {
            int thickness = std::max(1, (int)(i * 2 / t.trajectory.size()));
            cv::line(frame, t.trajectory[i - 1], t.trajectory[i], color, thickness);
        }
    }
}
```

在 `run()` 循环中，追踪模式下调用：

```cpp
if (trackingEnabled_) {
    auto tracks = tracker_.update(dets);
    drawTracks(currentFrame_, tracks);
    if (trajectoryEnabled_)
        drawTrajectory(currentFrame_, tracks);
}
```

#### 3. MainWindow 层 — UI 控制

- 工具栏新增可勾选按钮 `trajectoryBtn_`
- 快捷键设为 `Shift+T`（避免与现有 T 键冲突）
- 设置持久化：`settings.setValue("trajectory", enabled)`
- 切换输入源或关闭追踪时，轨迹自动清空

#### 4. Lang 字符串

```
{"trajectory",        "显示轨迹",           "Trajectory"},
{"trajectory_on",     "隐藏轨迹",           "Hide Trajectory"},
{"trajectory_off",    "显示轨迹",           "Trajectory"},
{"tip_trajectory",    "显示运动轨迹",        "Show motion trajectory"},
```

### 验收标准
- 开启后每个追踪目标显示彩色运动轨迹线
- 轨迹随目标移动实时更新
- 关闭追踪时轨迹消失，不泄漏内存
- 轨迹长度上限 200 点，旧点自动淘汰

---

## 功能二：越线计数

### 目标
用户在画面上画一条虚拟线，统计目标从一侧穿越到另一侧的次数，实现人流/车流统计。

### 实现方案

#### 1. 数据模型

**文件：`tracker.h`**

```cpp
struct CountingLine {
    cv::Point pt1, pt2;       // 线段两端点
    std::string label;         // 标签（如"入口A"）
};

struct CrossingEvent {
    int trackId;
    int classId;
    int direction;             // +1 正向穿越, -1 反向穿越
    int64_t timestampMs;       // 穿越时刻
};
```

InternalTrack 新增：

```cpp
struct InternalTrack {
    // ...existing fields...
    int lastSide;              // 上帧所在线段的侧（+1/-1/0）
};
```

#### 2. Tracker 层 — 越线检测

**文件：`tracker.h/cpp`**

```cpp
class Tracker {
    // ...existing...
    CountingLine countLine_;
    bool hasLine_ = false;
    std::vector<CrossingEvent> crossings_;
    QMap<int, int> crossingCounts_;  // classId -> count

public:
    void setCountingLine(const CountingLine& line);
    void clearCountingLine();
    bool hasCountingLine() const;
    QMap<int, int> crossingCounts() const;
    std::vector<CrossingEvent> recentCrossings(int lastN = 50) const;
};
```

越线检测算法：在 `update()` 中，对匹配成功的轨迹，计算当前帧中心点相对线段的侧（叉积符号），若侧发生翻转则判定穿越。

```cpp
static int sideOfLine(const cv::Point& p, const CountingLine& line) {
    return (line.pt2.x - line.pt1.x) * (p.y - line.pt1.y)
         - (line.pt2.y - line.pt1.y) * (p.x - line.pt1.x) > 0 ? 1 : -1;
}
```

#### 3. InferenceThread 层 — 绘制计数线

**文件：`inferencethread.h/cpp`**

```cpp
void drawCountingLine(cv::Mat& frame);  // 绘制虚线 + 标签 + 计数
```

在 `run()` 循环中调用绘制。越线瞬间可闪烁计数数字。

#### 4. MainWindow 层 — 交互式画线 + 统计面板

**交互式画线：**
- 工具栏按钮 `countLineBtn_`，点击后进入画线模式
- 画线模式下，鼠标在 videoLabel_ 上点击两点定义线段
- 弹出 QInputDialog 输入标签名
- 线段数据通过 InferenceThread 接口传递

**统计 Dock 窗口：**
- 新增 `countingDock_`，表格显示每个类别的越线次数
- 列：类别 | 正向 | 反向 | 合计
- 清零按钮

#### 5. Lang 字符串

```
{"counting_line",     "越线计数",           "Line Crossing"},
{"draw_line",         "画计数线",           "Draw Line"},
{"clear_line",        "清除计数线",         "Clear Line"},
{"line_label_prompt", "输入线段标签:",       "Enter line label:"},
{"crossing_count",    "越线次数",           "Crossing Count"},
```

### 验收标准
- 用户可在画面上画一条线，线段两端点可拖拽
- 目标穿越线段时计数 +1，区分正反方向
- 统计面板实时显示各类别越线次数
- 支持清除线段重新画

---

## 功能三：唯一目标计数

### 目标
统计每个类别出现的独立目标数量（基于 trackId），区别于当前的检测次数累加。

### 实现方案

#### 1. Tracker 层 — 统计唯一 ID

**文件：`tracker.h/cpp`**

```cpp
class Tracker {
    // ...existing...
    QSet<int> seenIds_;                      // 所有出现过的 trackId
    QMap<int, QSet<int>> seenIdsByClass_;    // classId -> 该类别出现过的 trackId 集合

public:
    QMap<int, int> uniqueCounts() const;     // classId -> 唯一目标数
    int totalUnique() const;
    void resetCounts();                       // 清零（不清空追踪状态）
};
```

在 `update()` 中维护：新轨迹创建时记录 trackId，已有轨迹不重复计数。

#### 2. InferenceThread 层 — 传递统计数据

在 `frameReady` 信号中扩展参数，或新增独立信号：

```cpp
// 方案A：扩展 frameReady（需修改信号签名，影响已有 connect）
// 方案B：新增信号（推荐，不影响已有接口）
signals:
    void statsUpdated(const QMap<int,int>& uniqueCounts, int totalUnique);
```

#### 3. MainWindow 层 — 统计面板增强

**修改现有 statsDock_：**
- 表格新增一列"唯一目标数"
- 列：类别 | 检测次数 | 唯一目标数
- 底部显示合计

**新增 resetCounts 按钮**（区别于现有的 clearStats，仅清零唯一计数）。

#### 4. Lang 字符串

```
{"unique_count",      "唯一目标",           "Unique"},
{"total_unique",      "总计: %1",           "Total: %1"},
{"reset_counts",      "重置计数",           "Reset Counts"},
```

### 验收标准
- 统计面板新增"唯一目标"列
- 同一个 trackId 只计一次，即使连续出现多帧
- 区别于检测次数（同一目标被检测 100 帧计为 1 个唯一目标）
- 重置按钮可清零计数

---

## 功能四：速度与方向估算

### 目标
基于 Kalman 滤波状态估算目标的速度（像素/帧 或 换算为 m/s）和运动方向，显示在标签上。

### 实现方案

#### 1. Tracker 层 — 提取速度状态

**文件：`tracker.h`**

```cpp
struct Track {
    Detection det;
    int trackId;
    std::vector<cv::Point> trajectory;
    float speed;              // 新增：像素/帧
    float angle;              // 新增：运动方向角度（0-360°）
};
```

**文件：`tracker.cpp`**

Kalman 滤波器已包含速度状态（`vcx`, `vcy`），在 `update()` 中提取：

```cpp
// 在更新匹配轨迹后
float vx = tracks_[i].kf.statePost.at<float>(4);
float vy = tracks_[i].kf.statePost.at<float>(5);
tracks_[i].speed = std::sqrt(vx * vx + vy * vy);
tracks_[i].angle = std::atan2(vy, vx) * 180.f / CV_PI;
if (tracks_[i].angle < 0) tracks_[i].angle += 360.f;
```

#### 2. InferenceThread 层 — 可视化

**文件：`inferencethread.cpp`**

修改 `drawTracks()` 标签格式：

```
#3 car 0.87  12.5px/f →
```

可选：在目标中心绘制方向箭头：

```cpp
cv::arrowedLine(frame, center, endPoint, color, 2);
```

#### 3. MainWindow 层 — UI 控制

- 工具栏新增可勾选按钮 `speedBtn_`
- 开启后标签显示速度和方向，目标中心绘制箭头
- 设置持久化

#### 4. Lang 字符串

```
{"show_speed",        "速度方向",           "Speed"},
{"speed_unit",        "px/f",              "px/f"},
```

### 验收标准
- 开启后追踪标签显示速度值和方向箭头
- 速度值平滑（基于 Kalman 状态而非帧间差分，避免抖动）
- 目标静止时速度接近 0

---

## 功能五：追踪数据导出

### 目标
将追踪历史数据（轨迹、越线事件、唯一计数）导出为 CSV/JSON 文件，供后续分析。

### 实现方案

#### 1. InferenceThread 层 — 数据采集

**文件：`inferencethread.h`**

```cpp
struct TrackRecord {
    int trackId;
    int classId;
    int64_t timestampMs;
    int x, y, width, height;
    float speed, angle;
};

// 新增接口
std::vector<TrackRecord> trackHistory() const;
void clearTrackHistory();
```

**文件：`inferencethread.cpp`**

在 `run()` 循环中，追踪模式下记录每帧的 Track 数据到 `trackHistory_`（带 mutex 保护），可设置最大记录条数限制内存。

#### 2. MainWindow 层 — 扩展导出功能

**修改 `onExport()` 方法：**

当追踪开启时，导出对话框增加选项：
- 检测结果（现有）
- 追踪轨迹（新增）
- 越线事件（新增，依赖功能二）

使用 `QFileDialog` 的文件类型过滤区分：
- `*.csv` / `*.json` — 检测结果（现有）
- `*_tracks.csv` / `*_tracks.json` — 追踪轨迹

**导出格式：**

CSV 轨迹：
```
trackId,class,timestamp,x,y,width,height,speed,angle
3,car,20260428_143012_500,120,80,60,40,12.5,45.0
```

JSON 轨迹：
```json
{
  "tracks": [
    {
      "trackId": 3,
      "class": "car",
      "points": [
        {"t": "2026-04-28T14:30:12.500", "x": 120, "y": 80, "w": 60, "h": 40, "speed": 12.5, "angle": 45.0},
        ...
      ]
    }
  ],
  "uniqueCounts": {"car": 5, "person": 3},
  "crossings": [...]
}
```

#### 3. Lang 字符串

```
{"export_tracks",     "导出追踪数据",         "Export Tracking Data"},
{"export_track_filter","追踪数据 (*.json *.csv)", "Tracking Data (*.json *.csv)"},
{"export_track_done", "追踪数据已导出: ",      "Tracking data exported: "},
```

### 验收标准
- 导出 CSV/JSON 包含完整追踪轨迹数据
- 每个目标包含 ID、类别、时间戳、坐标、速度、方向
- 大文件不卡 UI（异步写入或限制记录条数）

---

## 开发优先级建议

| 优先级 | 功能 | 复杂度 | 依赖 |
|--------|------|--------|------|
| P0 | 功能三：唯一目标计数 | 低 | 无 |
| P0 | 功能一：轨迹线可视化 | 低 | 无 |
| P1 | 功能五：追踪数据导出 | 中 | 功能一、三 |
| P2 | 功能四：速度与方向估算 | 低 | 无 |
| P2 | 功能二：越线计数 | 高 | 功能一 |

建议按 P0 → P1 → P2 顺序开发，功能二（越线计数）复杂度最高且需要交互式画线 UI，建议最后实现。
