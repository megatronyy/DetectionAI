# Plan: UI 布局重构 — 菜单栏 + 精简工具栏 + 标签化面板

## Context

当前 MainWindow 的单个工具栏承载了 22 个按钮 + 摄像头下拉框，过于拥挤。5 个独立 QDockWidget 堆叠在右侧，各有标题栏浪费空间。菜单栏只有"关于"，大量功能只能通过工具栏按钮访问。

**目标**：通过菜单栏分组 + 精简工具栏 + 统一标签面板，让界面清晰有序，所有功能仍然可访问。

## 当前问题

| 问题 | 详情 |
|------|------|
| 工具栏拥挤 | 22 个按钮 + 1 个下拉框挤在一行 |
| 无菜单导航 | 除"关于"外无菜单，功能全靠按钮 |
| 5 个独立 Dock | 各有标题栏，占用空间，tabify 后仍杂乱 |
| 快捷键不可见 | 15+ 快捷键仅在 tooltip 中提及 |
| 条件按钮混乱 | 骨骼/双目/标定按钮根据状态出现/消失 |

## 重构方案：三部分改造

### 1. 完整菜单栏（替代大部分工具栏按钮）

```
文件(F)          模型(M)         播放(P)         追踪(T)          双目(B)         视图(V)        帮助(H)
─────────────   ─────────────   ─────────────   ──────────────   ──────────────  ─────────────  ─────────
打开视频  Ctrl+O 切换模型 Ctrl+M 暂停/继续 Space  ✓目标追踪      ✓双目模式       ✓全屏    F11    关于
网络摄像头 Ctrl+N 最近模型       截图       S    ✓轨迹线        ✓深度叠加       ─────────────
─────────────   ─────────────   录制       R    ✓速度方向      ─────────────    切换语言
摄像头 [子菜单]  类别筛选        ✓循环播放  L    画计数线    C  标定 Shift+B     显示面板
─────────────                   ─────────────   清除计数线      双目设置
导出     Ctrl+E
─────────────
退出     Ctrl+Q
```

**关键**：
- 所有 QAction 带 `setShortcut()`，Qt 自动处理快捷键分发
- Checkable actions（追踪、轨迹、速度、骨骼、双目、深度叠加、循环、全屏）用 `setCheckable(true)`
- 条件菜单项：骨骼（姿态模型）、标定/深度/双目设置（双目模式）通过 `setVisible()` 控制
- 最近模型：动态子菜单

### 2. 精简工具栏（仅 6 项）

```
| [暂停] [截图] [录制] [导出] | 输入: [摄像头下拉] | [中/EN] |
```

- 6 个操作直接用 `toolbar->addAction(actXxx_)` 从菜单共享 QAction
- 移除 16 个不常用按钮（模型切换、类别筛选、追踪/轨迹/速度、骨骼、双目、标定、计数线等全部归入菜单）

### 3. 统一右侧标签面板（替代 5 个 Dock）

用一个 QDockWidget + QTabWidget 替代 5 个独立 QDockWidget：

```
┌─────────────────────────┐
│ 统计 | 追踪 | 姿态 | 深度 │  ← QTabWidget
├─────────────────────────┤
│ [追踪✓] [轨迹] [速度]    │  ← 标签页内的切换按钮（追踪 tab）
│                         │
│  统计表格 / 越线表格 /   │
│  关键点表格 / 深度表格    │
│                         │
└─────────────────────────┘
```

| Tab | 内容 | 可见条件 |
|-----|------|----------|
| 统计 | 统计表 + 唯一计数 + 清零/重置按钮 | 始终 |
| 追踪 | 越线表 + 画线/清除按钮 + 追踪/轨迹/速度切换 | 始终 |
| 姿态 | 关键点表 + 关键点阈值滑块 | 姿态模型 |
| 深度 | 深度表 + 点云视图 + 深度叠加切换 | 双目模式 |

## 修改文件

| 文件 | 变更 |
|------|------|
| `mainwindow.h` | 删除 22 个 QPushButton* 成员 → 换为 QAction* + QMenu*；删除 5 个 QDockWidget* → 换为 1 个 panelDock_ + QTabWidget*；新增 createAction() 辅助方法 |
| `mainwindow.cpp` | 重写 setupUI()（菜单 + 精简工具栏 + 标签面板）、refreshUIText()、updateModelTypeUI()、keyPressEvent()（仅保留 Esc）、loadSettings()/saveSettings()（panelVisible + panelTab 索引） |
| `lang.cpp` | 新增 ~15 个菜单标题/退出/全屏等 Lang 字符串 |

**不修改**：InferenceThread、YOLODetector、Tracker、StereoSource、StereoMatcher、所有 Dialog 类、DetectionAI.pro

## 成员变量变更明细

### 删除（mainwindow.h）

```cpp
// 22 个按钮指针
QPushButton* pauseBtn_; screenshotBtn_; recordBtn_; exportBtn_;
QPushButton* videoBtn_; networkCamBtn_; loopBtn_;
QPushButton* switchModelBtn_; recentModelBtn_; classFilterBtn_;
QPushButton* trackingBtn_; trajectoryBtn_; speedBtn_; skeletonBtn_;
QPushButton* stereoBtn_; calibrateBtn_; depthOverlayBtn_; stereoSettingsBtn_;
QPushButton* countLineBtn_; clearLineBtn_; langBtn_;

// 5 个独立 Dock
QDockWidget* statsDock_; QDockWidget* countingDock_;
QDockWidget* poseDock_; QDockWidget* depthDock_; QDockWidget* pointCloudDock_;

// 部分面板内按钮
QPushButton* clearStatsBtn_; QPushButton* resetCountsBtn_;
QPushButton* clearCrossingBtn_;
```

### 新增（mainwindow.h）

```cpp
// 菜单
QMenu* fileMenu_; QMenu* modelMenu_; QMenu* playbackMenu_;
QMenu* trackingMenu_; QMenu* stereoMenu_; QMenu* viewMenu_; QMenu* helpMenu_;
QMenu* recentModelsMenu_;

// Actions（替代所有按钮）
QAction* actPause_; QAction* actScreenshot_; QAction* actRecord_;
QAction* actExport_; QAction* actOpenVideo_; QAction* actNetworkCam_;
QAction* actLoop_; QAction* actSwitchModel_; QAction* actClassFilter_;
QAction* actTracking_; QAction* actTrajectory_; QAction* actSpeed_;
QAction* actSkeleton_; QAction* actStereo_; QAction* actDepthOverlay_;
QAction* actCalibrate_; QAction* actStereoSettings_;
QAction* actCountLine_; QAction* actClearLine_;
QAction* actLanguage_; QAction* actFullScreen_; QAction* actExit_;

// 统一面板
QDockWidget* panelDock_;
QTabWidget* panelTabs_;
QWidget* statsPage_;
QWidget* trackingPage_;
QWidget* posePage_;
QWidget* depthPage_;

// 辅助
QAction* createAction(const QString& text, const QString& tip,
                      const QKeySequence& shortcut = QKeySequence(),
                      bool checkable = false, bool checked = false);
```

### 保留不变

```cpp
QLabel* videoLabel_;
QComboBox* cameraCombo_;
QSlider* confSlider_; QLabel* confValueLabel_;
QSlider* iouSlider_; QLabel* iouValueLabel_;
QLabel* fpsLabel_; QLabel* detLabel_; QLabel* inferLabel_; QLabel* deviceLabel_;
QTableWidget* statsTable_; QTableWidget* countingTable_;
QTableWidget* poseTable_; QTableWidget* depthTable_;
QLabel* posePersonLabel_; QSlider* kpConfSlider_; QLabel* kpConfValueLabel_;
QLabel* pointCloudLabel_;
QLabel* totalUniqueLabel_;
// 所有状态成员不变
```

## 实施步骤（9 步，每步可编译测试）

### Step 1: mainwindow.h — 添加新成员
- 添加 QAction*、QMenu*、QTabWidget* 成员
- 添加 createAction() 声明
- 暂不删除旧 QPushButton* 成员（渐进迁移）

### Step 2: mainwindow.cpp — 创建 createAction() 辅助函数
```cpp
QAction* MainWindow::createAction(const QString& text, const QString& tip,
                                   const QKeySequence& shortcut,
                                   bool checkable, bool checked)
{
    QAction* a = new QAction(text, this);
    a->setToolTip(tip);
    if (!shortcut.isEmpty()) a->setShortcut(shortcut);
    a->setCheckable(checkable);
    if (checkable) a->setChecked(checked);
    a->setShortcutContext(Qt::ApplicationShortcut);
    return a;
}
```

### Step 3: mainwindow.cpp — 重写 setupUI()
- 创建所有 QAction（带快捷键和 checkable 状态）
- 构建完整菜单栏（7 个菜单）
- 精简工具栏（6 个 addAction + cameraCombo）
- 创建 panelDock_ + panelTabs_，4 个标签页
- 迁移各 Dock 内容到对应 Tab 页
- 移除旧 5 个 Dock 的创建代码
- 连接所有信号

### Step 4: mainwindow.cpp — 更新构造函数
- 信号连接保持不变（frameReady、inputLost、trackingStatsUpdated 等）
- depthMapReady 信号连接不变

### Step 5: mainwindow.cpp — 重写 loadSettings()/saveSettings()
- 旧设置键（statsVisible、countingDockVisible 等）迁移为 panelVisible + panelTab
- 用 actXxx_->setChecked() 替代 btnXxx_->setChecked()

### Step 6: mainwindow.cpp — 重写 refreshUIText()
- 更新 7 个菜单标题
- 更新 ~20 个 QAction 文字
- 更新 4 个 Tab 标题
- 更新表格表头、状态栏等

### Step 7: mainwindow.cpp — 精简 keyPressEvent()
- 仅保留 Qt::Key_Escape（画线取消 / 退出全屏 / 关闭窗口）
- 其他所有快捷键由 QAction::setShortcut() 自动处理

### Step 8: lang.cpp — 新增菜单相关字符串
新增 ~15 个 Lang key：
- menu_file, menu_model, menu_playback, menu_tracking, menu_stereo, menu_view, menu_help
- menu_exit, menu_fullscreen, menu_show_panel 等

### Step 9: mainwindow.h — 删除旧成员
- 删除 22 个 QPushButton* 成员
- 删除 5 个旧 QDockWidget* 成员
- 删除 clearStatsBtn_、resetCountsBtn_、clearCrossingBtn_（移入 Tab 页局部变量或保留为成员）

## 风险与应对

| 风险 | 应对 |
|------|------|
| Space 快捷键与 QComboBox 冲突 | QAction 设 `setShortcutContext(Qt::ApplicationShortcut)` |
| QTabWidget::setTabVisible() 隐藏所有 Tab | 始终保持"统计"Tab 可见 |
| 旧设置迁移 | loadSettings() 检测 panelVisible 不存在时从旧 key 推算 |
| 最近模型子菜单需动态更新 | addRecentModel() 时重建子菜单 |
| refreshUIText() 行数多 | 可考虑数据驱动数组，但暂用手动更新 |

## 验证

1. 启动 → 工具栏仅 6 项，菜单栏 7 个菜单
2. 快捷键 Space/S/O/N/M/T/L/C/E 等全部正常
3. 菜单中 checkable 项状态与功能同步
4. 右侧面板 4 个 Tab 切换正常，姿态/深度 Tab 条件显示
5. 设置保存/恢复正常（面板可见性、选中 Tab、追踪状态等）
6. 画线模式 Escape 取消正常
7. 模型切换后 UI 更新正确（检测/姿态按钮、Tab 可见性）
8. 双目模式开关后 UI 更新正确
