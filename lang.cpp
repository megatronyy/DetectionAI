#include "lang.h"

Lang::Language Lang::lang_ = Chinese;

void Lang::setLanguage(Language lang) { lang_ = lang; }
Lang::Language Lang::language() { return lang_; }

static const struct { const char* key; const char* zh; const char* en; } strings[] = {
    // Window
    {"app_title",       "YOLO11 物体检测",           "YOLO11 Object Detection"},

    // Toolbar buttons
    {"pause",           "暂停",                      "Pause"},
    {"resume",          "继续",                      "Resume"},
    {"screenshot",      "截屏",                      "Screenshot"},
    {"record",          "录制",                      "Record"},
    {"stop_record",     "停止录制",                   "Stop Recording"},
    {"open_video",      "打开视频",                   "Open Video"},
    {"network_cam",     "网络摄像头",                  "Network Camera"},
    {"loop",            "循环播放",                   "Loop Playback"},
    {"switch_model",    "切换模型",                   "Switch Model"},
    {"class_filter",    "类别筛选",                   "Class Filter"},
    {"tracking_on",     "关闭追踪",                   "Tracking On"},
    {"tracking_off",    "目标追踪",                   "Tracking"},
    {"export_btn",      "导出",                      "Export"},
    {"lang_toggle",     "EN",                        "中文"},

    // Slider labels
    {"confidence",      "置信度:",                    "Confidence:"},
    {"iou",             "IoU:",                      "IoU:"},

    // Status bar
    {"fps",             "FPS: %1",                   "FPS: %1"},
    {"det_count",       "检测数: %1",                 "Detections: %1"},
    {"infer_ms",        "推理: %1ms",                 "Infer: %1ms"},
    {"device_cpu",      "CPU",                       "CPU"},
    {"device_gpu",      "GPU (CUDA)",                "GPU (CUDA)"},

    // Input label
    {"input_source",    " 输入: ",                    " Input: "},
    {"camera",          "摄像头 %1",                  "Camera %1"},

    // Dialogs - model
    {"select_model",    "选择 ONNX 模型",             "Select ONNX Model"},
    {"model_filter",    "ONNX 模型 (*.onnx)",         "ONNX Models (*.onnx)"},
    {"model_load_fail", "无法加载 ONNX 模型。",        "Failed to load ONNX model."},
    {"model_switch_fail","无法加载模型: ",             "Failed to load model: "},
    {"model_switched",  "模型已切换: ",                "Model switched: "},

    // Dialogs - camera
    {"cam_warn",        "警告",                       "Warning"},
    {"cam_open_fail",   "无法打开默认摄像头，请选择其他输入源或打开视频文件。",
     "Cannot open default camera. Please select another input source or open a video file."},
    {"cam_switch_fail", "无法打开摄像头 %1。",          "Cannot open camera %1."},
    {"cam_switched",    "已切换到摄像头 %1",            "Switched to camera %1"},
    {"cam_disconnected","摄像头已断开",                 "Camera disconnected"},

    // Dialogs - video
    {"open_video_title","打开视频",                    "Open Video"},
    {"video_filter",    "视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv);;所有文件 (*)",
     "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*)"},
    {"video_open_fail", "无法打开视频文件。",            "Cannot open video file."},
    {"video_opened",    "视频: ",                      "Video: "},
    {"video_ended",     "视频播放结束",                  "Video playback ended"},

    // Dialogs - network
    {"network_title",   "网络摄像头",                   "Network Camera"},
    {"network_prompt",  "输入 RTSP/HTTP 视频 URL:",     "Enter RTSP/HTTP video URL:"},
    {"network_fail",    "无法打开网络视频流。\n请检查 URL 是否正确以及网络连接。",
     "Cannot open network stream.\nPlease check the URL and network connection."},
    {"network_opened",  "网络流: ",                     "Stream: "},

    // Dialogs - screenshot
    {"save_screenshot", "保存截图",                     "Save Screenshot"},
    {"image_filter",    "图片 (*.png *.jpg *.bmp)",     "Images (*.png *.jpg *.bmp)"},
    {"screenshot_saved","截图已保存: ",                  "Screenshot saved: "},
    {"screenshot_fail", "保存截图失败。",                "Failed to save screenshot."},

    // Dialogs - recording
    {"save_recording",  "保存录制",                     "Save Recording"},
    {"video_save_filter","视频 (*.mp4 *.avi)",          "Video (*.mp4 *.avi)"},
    {"recording_stopped","录制已停止",                   "Recording stopped"},
    {"recording",       "录制中...",                    "Recording..."},
    {"recording_fail",  "无法创建录制文件。",             "Failed to create recording file."},

    // Tracking
    {"tracking_enabled","目标追踪已启用",                "Object tracking enabled"},
    {"tracking_disabled","目标追踪已关闭",               "Object tracking disabled"},

    // Class filter dialog
    {"class_filter_title","类别筛选",                   "Class Filter"},
    {"select_all",      "全选",                        "Select All"},
    {"select_none",     "全不选",                       "Select None"},

    // Statistics dock
    {"stats_title",     "检测统计",                     "Detection Statistics"},
    {"stats_class",     "类别",                        "Class"},
    {"stats_count",     "次数",                        "Count"},
    {"stats_unique",    "唯一",                        "Unique"},
    {"stats_total_unique", "唯一目标总计: %1",          "Total unique: %1"},

    // Export
    {"export_title",    "导出检测结果",                  "Export Detections"},
    {"export_filter",   "JSON (*.json);;CSV (*.csv)",  "JSON (*.json);;CSV (*.csv)"},
    {"export_done",     "检测结果已导出: ",              "Detections exported: "},
    {"export_fail",     "导出检测结果失败。",             "Failed to export detections."},
    {"export_no_data",  "当前无检测结果可导出。",          "No detections to export."},

    // Tracking export
    {"export_tracks",     "导出追踪数据",               "Export Tracking Data"},
    {"export_track_filter","追踪数据 (*.json *.csv)",    "Tracking Data (*.json *.csv)"},
    {"export_track_done", "追踪数据已导出: ",            "Tracking data exported: "},
    {"export_track_no_data","当前无追踪数据可导出。",     "No tracking data to export."},
    {"export_choice",     "选择导出内容",               "Choose Export Content"},
    {"export_detect",     "检测结果",                   "Detections"},
    {"export_tracking",   "追踪数据",                   "Tracking Data"},

    // Errors
    {"error",           "错误",                        "Error"},

    // Loop playback
    {"loop_enabled",    "循环播放已启用",                "Loop playback enabled"},
    {"loop_disabled",   "循环播放已关闭",                "Loop playback disabled"},

    // About dialog
    {"about",           "关于",                        "About"},
    {"about_text",      "<h2>DetectionAI</h2>"
                        "<p>YOLO11 实时目标检测</p>"
                        "<p>基于 Qt %1 / OpenCV %2 / ONNX Runtime</p>"
                        "<p>COCO 80 类 | SORT 多目标追踪</p>",
     "<h2>DetectionAI</h2>"
     "<p>YOLO11 Real-time Object Detection</p>"
     "<p>Powered by Qt %1 / OpenCV %2 / ONNX Runtime</p>"
     "<p>COCO 80 Classes | SORT Multi-Object Tracking</p>"},

    // Tooltips
    {"tip_pause",       "暂停/继续 (Space)",             "Pause/Resume (Space)"},
    {"tip_screenshot",  "截图 (S)",                     "Screenshot (S)"},
    {"tip_record",      "录制视频",                      "Record video"},
    {"tip_export",      "导出检测结果 (E)",              "Export detections (E)"},
    {"tip_open_video",  "打开视频文件 (O)",              "Open video file (O)"},
    {"tip_network",     "网络摄像头 (N)",                "Network camera (N)"},
    {"tip_loop",        "循环播放 (L)",                  "Loop playback (L)"},
    {"tip_model",       "切换模型 (M)",                  "Switch model (M)"},
    {"tip_filter",      "类别筛选",                     "Class filter"},
    {"tip_tracking",    "目标追踪 (T)",                  "Object tracking (T)"},
    {"tip_lang",        "切换语言",                      "Switch language"},

    // Drag & drop
    {"drop_video",      "已打开: ",                     "Opened: "},
    {"drop_fail",       "无法打开拖入的文件。",           "Cannot open dropped file."},
    {"drop_model_ok",   "模型已加载: ",                  "Model loaded: "},
    {"drop_model_fail", "无法加载拖入的模型文件。",       "Cannot load dropped model file."},

    // Recent models
    {"recent_models",   "最近模型",                     "Recent Models"},

    // Stats clear
    {"stats_clear",     "清零统计",                     "Clear Stats"},
    {"reset_counts",    "重置计数",                     "Reset Counts"},

    // Trajectory
    {"trajectory_on",   "隐藏轨迹",                     "Hide Trail"},
    {"trajectory_off",  "显示轨迹",                     "Trail"},
    {"tip_trajectory",  "显示运动轨迹",                  "Show motion trail"},

    // Speed & direction
    {"speed_on",        "隐藏速度",                     "Hide Speed"},
    {"speed_off",       "速度方向",                     "Speed"},
    {"tip_speed",       "显示速度与方向 (Shift+S)",      "Show speed & direction (Shift+S)"},

    // Line crossing counting
    {"counting_line",   "越线计数",                     "Line Crossing"},
    {"draw_line",       "画计数线",                     "Draw Line"},
    {"clear_line",      "清除计数线",                   "Clear Line"},
    {"line_label_prompt","输入线段标签:",                "Enter line label:"},
    {"crossing_count",  "越线统计",                     "Crossing Count"},
    {"forward",         "正向",                        "Forward"},
    {"reverse",         "反向",                        "Reverse"},
    {"total",           "合计",                        "Total"},
    {"click_pt1",       "点击画面设置第一个点",           "Click to set first point"},
    {"click_pt2",       "点击画面设置第二个点",           "Click to set second point"},
    {"line_set",        "计数线已设置",                  "Counting line set"},
    {"line_cleared",    "计数线已清除",                  "Counting line cleared"},
    {"draw_cancelled",  "画线已取消",                    "Drawing cancelled"},
    {"tip_draw_line",   "在画面上画一条计数线 (C)",       "Draw a counting line (C)"},
    {"tip_clear_line",  "清除计数线并重置计数",            "Clear line and reset counts"},

    // Pose estimation
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

    // Stereo / binocular vision
    {"stereo_mode",       "双目模式",              "Stereo Mode"},
    {"stereo_on",         "关闭双目",              "Stereo On"},
    {"stereo_off",        "双目模式",              "Stereo"},
    {"tip_stereo",        "双目立体视觉 (B)",       "Stereo vision (B)"},
    {"stereo_settings",   "双目设置",              "Stereo Settings"},
    {"stereo_hw",         "硬件类型",              "Hardware"},
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
    {"stereo_connected",  "双目设备已连接",          "Stereo device connected"},
    {"stereo_disconnected","双目设备断开",          "Stereo device disconnected"},

    // Calibration
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

    // Depth / SGBM
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

    // Depth dock columns
    {"depth_track_id",     "追踪ID",                 "Track ID"},
    {"depth_class",        "类别",                   "Class"},
    {"depth_dist",         "距离(m)",                "Dist (m)"},
    {"depth_conf",         "深度置信度",              "Depth Conf"},

    // Point cloud
    {"point_cloud",        "点云视图",               "Point Cloud"},
    {"point_cloud_dock",   "鸟瞰点云",               "Bird's Eye View"},

    // Export
    {"export_pointcloud",     "导出点云",                    "Export Point Cloud"},
    {"export_pc_filter",      "点云 (*.ply *.xyz)",          "Point Cloud (*.ply *.xyz)"},
    {"export_pc_done",        "点云已导出: ",                 "Point cloud exported: "},

    // Menu bar
    {"menu_file",            "文件",                        "File"},
    {"menu_model",           "模型",                        "Model"},
    {"menu_playback",        "播放",                        "Playback"},
    {"menu_tracking",        "追踪",                        "Tracking"},
    {"menu_stereo",          "双目",                        "Stereo"},
    {"menu_view",            "视图",                        "View"},
    {"menu_help",            "帮助",                        "Help"},
    {"menu_exit",            "退出",                        "Exit"},
    {"menu_fullscreen",      "全屏",                        "Full Screen"},

    {nullptr, nullptr, nullptr}
};

QString Lang::s(const QString& key)
{
    for (int i = 0; strings[i].key != nullptr; i++) {
        if (key == strings[i].key)
            return (lang_ == Chinese) ? QString::fromUtf8(strings[i].zh)
                                      : QString::fromUtf8(strings[i].en);
    }
    return key;
}
