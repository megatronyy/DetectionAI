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
