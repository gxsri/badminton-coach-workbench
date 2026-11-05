#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
yolo-pose-bridge.py —— 羽毛球动作视频的本地姿态解算 + 骨架分析视频生成

用途：把一段训练视频解算成 COCO-17 关键点序列 JSON，并可选直接输出
      「带骨架叠加的 MP4 分析视频」。
      JSON 供 coach-workbench.html 的「视频评估」导入，得到真实关节角度评分；
      MP4 可直接播放、分享、逐帧核对骨架是否贴合人体。

安装依赖（只需一次）：
    pip install opencv-python numpy

默认引擎 onnx 使用随项目提供的 yolo11n-pose.onnx（YOLO11n-pose，COCO-17），
由 OpenCV 自带 DNN 模块推理，不需要 torch、不需要 ultralytics、不需要联网下载权重。
这是「拿来就能产出骨架」的推荐路径；模型文件缺失时可执行 --fetch-model 重新获取。

基本用法：
    # 只产出 JSON
    python yolo-pose-bridge.py smash.mp4 -o smash_pose.json

    # 一次产出 JSON + 带骨架的 MP4（推荐）
    python yolo-pose-bridge.py smash.mp4 --overlay

    # 纯骨架背景（不叠原视频），便于看清动作链路
    python yolo-pose-bridge.py smash.mp4 --overlay --bg plain

    # 已有 JSON，不再跑模型，直接补出骨架视频（最快）
    python yolo-pose-bridge.py smash.mp4 --pose-json smash_pose.json

    # 指定模型 / 抽样 / 输出路径
    python yolo-pose-bridge.py smash.mp4 --overlay \
        --stride 1 --max-frames 300 --overlay-out smash_skeleton.mp4

    # 改用 ultralytics 引擎（需 pip install ultralytics，可选 GPU）
    python yolo-pose-bridge.py smash.mp4 --engine ultralytics \
        --model yolov8s-pose.pt --device 0 --overlay

参数说明：
    video              输入视频文件（mp4 / mov / avi 等，OpenCV 能读即可；
                       实测 H.265/HEVC 的微信视频可直接解码）
    -o/--out           输出 JSON 路径（默认：与视频同名 + _pose.json）
    --overlay          额外生成带骨架的 MP4（默认：<视频名>_skeleton.mp4）
    --overlay-out      显式指定 MP4 路径（隐含 --overlay）
    --bg               叠加背景：video=原视频（默认）/ plain=浅色纯骨架
    --pose-json        从已有姿态 JSON 生成骨架视频，跳过姿态推理
    --no-ids           骨架视频不显示关键点编号
    --codec            输出 fourcc，默认 mp4v；avi 可试 XVID
    --engine           auto（默认）/ onnx / ultralytics
    --model            模型文件；默认 onnx 用脚本同目录的 yolo11n-pose.onnx，
                       ultralytics 用 yolov8n-pose.pt
    --fetch-model      下载 ONNX 姿态模型到脚本同目录后退出
    --input-size       ONNX 推理输入边长（默认 640）
    --stride           每隔 N 帧处理一帧（默认 1，即逐帧）
    --max-frames       最多处理多少帧（默认 600，约 20 秒 @30fps）
    --conf             检测置信度阈值（默认 0.35，onnx 引擎）
    --device           仅 ultralytics 引擎的推理设备：cpu / 0（GPU）等

JSON 输出格式（与工作台约定一致）：
    {
      "video": "smash.mp4",
      "fps": 30.0,
      "width": 1280,
      "height": 720,
      "model": "yolov8n-pose.pt",
      "frames": [
        {"t": 0.0,  "kps": [[x, y, conf], ... 17 个点 ...]},
        {"t": 0.033,"kps": [[x, y, conf], ...]}
      ]
    }
    width / height 是原视频像素尺寸，工作台据此把骨架精确贴回画面。

COCO-17 关键点顺序：
    0 nose  1 L-Eye  2 R-Eye  3 L-Ear  4 R-Ear
    5 L-Shoulder  6 R-Shoulder  7 L-Elbow  8 R-Elbow  9 L-Wrist 10 R-Wrist
    11 L-Hip  12 R-Hip  13 L-Knee  14 R-Knee  15 L-Ankle 16 R-Ankle

拍摄建议：侧面或斜侧后 45°，全身入镜，机位固定，包含 3 次以上完整动作。

注意：全部计算在本机完成，视频不会上传到任何服务器。
"""

import argparse
import json
import os
import sys

# COCO-17 骨架连线（与工作台 coach-workbench.html 的 COCO_BONES 完全一致）
BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
LEFT = {1, 3, 5, 7, 9, 11, 13, 15}
RIGHT = {2, 4, 6, 8, 10, 12, 14, 16}
# BGR 颜色：左蓝 / 右红 / 中轴青
COLOR_LEFT = (235, 99, 37)
COLOR_RIGHT = (38, 38, 220)
COLOR_MID = (136, 148, 13)

# 默认 ONNX 姿态模型（YOLO11n-pose，COCO-17 输出，由 OpenCV DNN 直接推理）
ONNX_NAME = "yolo11n-pose.onnx"
# GitHub release asset：v8.3.0 / yolo11n-pose.onnx
ONNX_ASSET_API = ("https://api.github.com/repos/ultralytics/assets/"
                  "releases/assets/329009385")


def parse_args():
    ap = argparse.ArgumentParser(
        description="训练视频 → COCO-17 关键点 JSON + 骨架分析视频 MP4")
    ap.add_argument("video", nargs="?", default=None, help="输入视频文件路径")
    ap.add_argument("-o", "--out", default=None, help="输出 JSON 路径")
    ap.add_argument("--overlay", action="store_true", help="额外生成带骨架的 MP4")
    ap.add_argument("--overlay-out", default=None, help="MP4 输出路径（隐含 --overlay）")
    ap.add_argument("--bg", choices=["video", "plain"], default="video",
                    help="叠加原视频 / 浅色纯骨架背景")
    ap.add_argument("--pose-json", default=None, help="从已有姿态 JSON 生成骨架视频（跳过模型）")
    ap.add_argument("--no-ids", action="store_true", help="不显示关键点编号")
    ap.add_argument("--codec", default="mp4v", help="输出视频 fourcc（默认 mp4v）")
    ap.add_argument("--engine", choices=["auto", "onnx", "ultralytics"], default="auto",
                    help="姿态推理引擎（默认 auto：优先 ONNX / OpenCV DNN）")
    ap.add_argument("--model", default=None, help="模型文件（.onnx 或 .pt）")
    ap.add_argument("--fetch-model", action="store_true",
                    help="下载 ONNX 姿态模型到脚本同目录后退出")
    ap.add_argument("--input-size", type=int, default=640, help="ONNX 输入边长")
    ap.add_argument("--stride", type=int, default=1, help="每隔 N 帧处理一帧")
    ap.add_argument("--max-frames", type=int, default=600, help="最多处理帧数")
    ap.add_argument("--conf", type=float, default=0.35, help="检测置信度阈值")
    ap.add_argument("--device", default=None, help="ultralytics 推理设备，如 cpu / 0")
    return ap.parse_args()


def load_cv2():
    try:
        import cv2
    except ImportError:
        sys.exit("缺少依赖 opencv-python，请先执行：pip install opencv-python")
    return cv2


def open_video(path, cv2):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit("无法打开视频（编码不支持或文件损坏）：%s" % path)
    return cap


def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def default_onnx_path():
    return os.path.join(script_dir(), ONNX_NAME)


def fetch_onnx_model(dest=None):
    """通过 GitHub API 下载官方 YOLO11n-pose ONNX 模型（约 11.8 MB）。"""
    import urllib.request

    dest = dest or default_onnx_path()
    url = ONNX_ASSET_API
    req = urllib.request.Request(
        url, headers={"User-Agent": "AthleteIQ-PoseBridge",
                      "Accept": "application/octet-stream"})
    print("正在下载姿态模型 → %s" % dest)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = 0
        while True:
            chunk = r.read(262144)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    if total < 1000000:
        os.remove(dest)
        sys.exit("下载内容异常（%d 字节），已删除；请检查网络后重试。" % total)
    print("完成：%s（%.1f MB）" % (dest, total / 1e6))
    return dest


def load_onnx_net(cv2, path):
    """OpenCV 的 ONNX 读取器在 Windows 上无法处理含非 ASCII 字符的绝对路径
    （例如项目放在中文目录）。依次尝试：绝对路径 → 切到模型目录用相对名 →
    复制到纯 ASCII 临时目录。"""
    errors = []
    try:
        return cv2.dnn.readNetFromONNX(path)
    except Exception as e:
        errors.append(str(e))
    cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(os.path.abspath(path)))
        return cv2.dnn.readNetFromONNX(os.path.basename(path))
    except Exception as e:
        errors.append(str(e))
    finally:
        os.chdir(cwd)
    try:
        import shutil
        import tempfile
        tmp_dir = tempfile.gettempdir()
        tmp = os.path.join(tmp_dir, "bwb_pose_model.onnx")
        shutil.copyfile(path, tmp)
        return cv2.dnn.readNetFromONNX(tmp)
    except Exception as e:
        errors.append(str(e))
    raise RuntimeError("无法加载 ONNX 模型 %s：%s" % (path, " | ".join(errors)))


def resolve_engine(args):
    """决定推理引擎与模型路径；auto 优先使用随项目提供的 ONNX 模型。"""
    if args.engine == "ultralytics":
        return "ultralytics", (args.model or "yolov8n-pose.pt")
    if args.engine == "onnx":
        path = args.model or default_onnx_path()
        if not os.path.isfile(path):
            sys.exit("找不到 ONNX 模型：%s\n可执行 --fetch-model 下载，"
                     "或用 --model 指定本地 .onnx 文件。" % path)
        return "onnx", path
    # auto
    if args.model:
        return ("onnx", args.model) if args.model.lower().endswith(".onnx") \
            else ("ultralytics", args.model)
    if os.path.isfile(default_onnx_path()):
        return "onnx", default_onnx_path()
    return "ultralytics", "yolov8n-pose.pt"


class CvDnnPose:
    """用 OpenCV DNN 跑 YOLO-pose ONNX：无需 torch / ultralytics / onnxruntime。

    输出 [1, 56, 8400]：4 框 + 1 类别分 + 17×3 关键点，坐标在 letterbox
    输入图上；这里还原回原视频像素坐标，并返回 COCO-17 顺序的 (x, y, conf)。
    """

    def __init__(self, model_path, cv2, input_size=640, conf=0.35, iou=0.45):
        try:
            import numpy as np
        except ImportError:
            sys.exit("缺少依赖 numpy，请先执行：pip install numpy")
        self.np = np
        self.cv2 = cv2
        self.conf = float(conf)
        self.iou = float(iou)
        self.size = int(input_size)
        self.net = load_onnx_net(cv2, os.path.abspath(model_path))
        blob = np.zeros((1, 3, self.size, self.size), np.float32)
        self.net.setInput(blob)
        out = np.asarray(self.net.forward())
        if out.ndim != 3 or out.shape[1] < 56:
            raise RuntimeError("模型输出不符合 YOLO-pose 格式：%s" % (out.shape,))
        self.n_keypoints = (out.shape[1] - 5) // 3
        self.name = os.path.basename(model_path)

    def predict(self, frame, conf=None):
        np, cv2 = self.np, self.cv2
        thr = self.conf if conf is None else float(conf)
        H, W = frame.shape[:2]
        S = self.size
        scale = min(S / float(W), S / float(H))
        nw, nh = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
        pad_x, pad_y = (S - nw) / 2.0, (S - nh) / 2.0
        canvas = np.full((S, S, 3), 114, np.uint8)
        canvas[int(pad_y):int(pad_y) + nh, int(pad_x):int(pad_x) + nw] = \
            cv2.resize(frame, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (S, S),
                                     (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        pred = np.asarray(self.net.forward())[0].T  # [N, 56]

        scores = pred[:, 4]
        keep = np.where(scores > thr)[0]
        if not len(keep):
            return None
        rects, confs = [], []
        for i in keep:
            cx, cy, bw, bh = pred[i, :4]
            rects.append([float(cx - bw / 2), float(cy - bh / 2),
                          float(bw), float(bh)])
            confs.append(float(scores[i]))
        idx = cv2.dnn.NMSBoxes(rects, confs, thr, self.iou)
        if idx is None or len(idx) == 0:
            return None

        # 多人时优先「可见关键点多 + 框大」，与 ultralytics 分支口径一致
        best, best_score = None, -1.0
        for k in np.asarray(idx).flatten():
            i = keep[int(k)]
            kp = pred[i, 5:].reshape(self.n_keypoints, 3)
            visible = int((kp[:17, 2] > 0.15).sum())
            area = max(0.0, pred[i, 2]) * max(0.0, pred[i, 3])
            score = visible * 10000 + area
            if score > best_score:
                best_score, best = score, kp
        if best is None:
            return None

        out = []
        for j in range(17):
            if j >= len(best):
                out.append([0.0, 0.0, 0.0])
                continue
            x, y, c = best[j]
            if c <= 0.15:
                out.append([0.0, 0.0, 0.0])
            else:
                out.append([round(float((x - pad_x) / scale), 2),
                            round(float((y - pad_y) / scale), 2),
                            round(float(c), 3)])
        return out


def load_model(args, engine, model_path, cv2):
    if engine == "onnx":
        return CvDnnPose(model_path, cv2, args.input_size, args.conf)
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "缺少依赖 ultralytics，请先执行：\n"
            "    pip install ultralytics\n"
            "或改用默认 ONNX 引擎：python yolo-pose-bridge.py --fetch-model"
        )
    return YOLO(model_path)


def pick_main_person(result, min_visible=6):
    """当画面中有多个人时，选择「关键点可见数最多、且框最大」的那个。"""
    boxes = getattr(result, "boxes", None)
    kps = getattr(result, "keypoints", None)
    if kps is None or kps.xy is None or len(kps.xy) == 0:
        return None

    best, best_score = None, -1.0
    xy = kps.xy.cpu().tolist()
    conf = kps.conf.cpu().tolist() if kps.conf is not None else None

    for i, person in enumerate(xy):
        visible = 0
        if conf is not None and i < len(conf):
            visible = sum(1 for c in conf[i] if c > 0.15)
        area = 0.0
        if boxes is not None and boxes.xyxy is not None and i < len(boxes.xyxy):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().tolist()
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        score = visible * 10000 + area
        if visible >= min_visible and score > best_score:
            best_score, best = score, (person, conf[i] if conf is not None else None)
    if best is None and xy:
        # 兜底：取第一个人，避免整段完全丢失
        best = (xy[0], conf[0] if conf else None)
    return best


def bone_color(a, b):
    if a in LEFT and b in LEFT:
        return COLOR_LEFT
    if a in RIGHT and b in RIGHT:
        return COLOR_RIGHT
    return COLOR_MID


def draw_skeleton(cv2, img, kps, t, idx, model, trail, W, H, show_ids=True):
    """在一帧画面上绘制骨架、关键点、腕部轨迹与信息面板。"""
    line_aa = getattr(cv2, "LINE_AA", 16)
    font = getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0)

    def pt(i):
        if i >= len(kps):
            return None
        p = kps[i]
        if not p or len(p) < 2:
            return None
        c = p[2] if len(p) > 2 else 1.0
        if c <= 0.15:
            return None
        return (int(round(float(p[0]))), int(round(float(p[1]))))

    # 持拍侧腕点轨迹（最近 16 帧）
    for j in range(1, len(trail)):
        cv2.line(img, trail[j - 1], trail[j], (60, 90, 250), 2, line_aa)

    # 骨架连线：先白描边，再彩色描线，保证在任意背景上都清晰
    for a, b in BONES:
        A, B = pt(a), pt(b)
        if A is None or B is None:
            continue
        cv2.line(img, A, B, (255, 255, 255), 6, line_aa)
        cv2.line(img, A, B, bone_color(a, b), 3, line_aa)

    # 关键点与编号
    for i in range(17):
        P = pt(i)
        if P is None:
            continue
        cv2.circle(img, P, 4, bone_color(i, i), -1, line_aa)
        cv2.circle(img, P, 4, (255, 255, 255), 1, line_aa)
        if show_ids:
            cv2.putText(img, str(i), (P[0] + 6, P[1] - 6), font, 0.4,
                        (22, 32, 58), 1, line_aa)

    # 信息面板
    lines = [
        "AthleteIQ skeleton overlay  |  %s" % model,
        "t = %.2fs    frame = %d    points = 17" % (t, idx),
        "blue = left   red = right   teal = torso",
    ]
    bw = min(W - 24, 430)
    cv2.rectangle(img, (12, 12), (12 + bw, 12 + 18 * len(lines) + 10), (15, 20, 32), -1)
    for k, text in enumerate(lines):
        cv2.putText(img, text, (20, 32 + 18 * k), font, 0.48,
                    (245, 248, 252), 1, line_aa)


def skeleton_path(video_path):
    return os.path.splitext(video_path)[0] + "_skeleton.mp4"


def write_overlay(cv2, video_path, pose, out_path, bg="video",
                  codec="mp4v", show_ids=True):
    """按 pose.frames 的时间戳回读原视频帧，叠加骨架后写成 MP4。"""
    cap = open_video(video_path, cv2)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or pose.get("width") or 1280)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or pose.get("height") or 720)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or pose.get("fps") or 30.0
    stride = max(1, int(pose.get("stride") or 1))
    out_fps = float(pose.get("fps") or src_fps) / stride
    if out_fps <= 0:
        out_fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            "无法创建输出视频：%s。可试 --codec XVID 并把扩展名改成 .avi，"
            "或 --codec avc1（取决于本机编码器）" % out_path)

    frames = pose.get("frames") or []
    trail, written = [], 0
    print("正在生成骨架分析视频：%d 帧 → %s（%dx%d @ %.1f fps，背景=%s）"
          % (len(frames), out_path, W, H, out_fps, bg))

    # 顺序解码 + 按时间戳取帧：H.265/HEVC 随机 seek 慢且可能落在错误帧上，
    # 顺序读取既能保证骨架与原画面严格对齐，也适用于微信导出的视频。
    try:
        import numpy as np
    except ImportError:
        cap.release()
        writer.release()
        raise RuntimeError("骨架视频生成需要 numpy，请先执行：pip install numpy")
    targets = [int(round(float(f.get("t", idx * stride / src_fps)) * src_fps))
               for idx, f in enumerate(frames)]
    cur_frame, last_img, eof = 0, None, False

    for idx, f in enumerate(frames):
        t = float(f.get("t", idx * stride / src_fps))
        target = targets[idx]
        if bg == "video":
            while cur_frame <= target and not eof:
                ok, frame = cap.read()
                cur_frame += 1
                if not ok:
                    eof = True
                    break
                last_img = frame
            if last_img is None:
                continue
            img = last_img.copy()
        else:
            img = np.full((H, W, 3), (248, 250, 252), dtype=np.uint8)
            for gx in range(0, W, 40):
                cv2.line(img, (gx, 0), (gx, H), (226, 232, 240), 1)
            for gy in range(0, H, 40):
                cv2.line(img, (0, gy), (W, gy), (226, 232, 240), 1)

        kps = f.get("kps") or []
        wrist = kps[10] if len(kps) > 10 else None
        if wrist and len(wrist) >= 3 and wrist[2] > 0.15:
            trail.append((int(wrist[0]), int(wrist[1])))
            trail = trail[-16:]

        draw_skeleton(cv2, img, kps, t, idx,
                      str(pose.get("model") or "pose"), trail, W, H, show_ids)
        writer.write(img)
        written += 1
        if written % 30 == 0:
            sys.stdout.write("\r  已写入 %d / %d 帧" % (written, len(frames)))
            sys.stdout.flush()

    cap.release()
    writer.release()
    if written:
        print("\n完成：骨架分析视频 %d 帧 → %s" % (written, out_path))
    return written


def main():
    args = parse_args()
    if args.fetch_model:
        fetch_onnx_model()
        return
    if not args.video:
        sys.exit("请指定视频文件，例如：python yolo-pose-bridge.py 视频.mp4 --overlay")
    if not os.path.isfile(args.video):
        sys.exit("找不到视频文件：%s" % args.video)

    cv2 = load_cv2()
    want_overlay = args.overlay or bool(args.overlay_out) or bool(args.pose_json)

    # 模式 A：已有姿态 JSON，直接补出骨架视频（不跑模型）
    if args.pose_json:
        if not os.path.isfile(args.pose_json):
            sys.exit("找不到姿态 JSON：%s" % args.pose_json)
        with open(args.pose_json, "r", encoding="utf-8") as f:
            pose = json.load(f)
        out = args.overlay_out or skeleton_path(args.video)
        n = write_overlay(cv2, args.video, pose, out, args.bg, args.codec, not args.no_ids)
        if not n:
            sys.exit("未能写出任何帧：检查视频是否可解码、JSON 的 t 是否落在视频时长内。")
        print("接下来：把该 MP4 用于复盘，或在 coach-workbench.html「视频评估」导入 JSON 查看指标。")
        return

    # 模式 B：解算 JSON（可选同时产出骨架 MP4）
    engine, model_path = resolve_engine(args)
    cap = open_video(args.video, cv2)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    model = load_model(args, engine, model_path, cv2)
    model_name = os.path.basename(model_path)

    frames, idx, processed = [], 0, 0
    print("正在解算：%s（fps=%.1f，总帧数=%d，%dx%d）"
          % (os.path.basename(args.video), fps, total, W, H))
    print("引擎：%s　模型：%s　置信阈值：%.2f" % (engine, model_name, args.conf))

    while processed < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.stride != 0:
            idx += 1
            continue
        if engine == "onnx":
            kps = model.predict(frame, args.conf)
        else:
            results = model.predict(frame, conf=args.conf, verbose=False,
                                    device=args.device) if args.device else \
                      model.predict(frame, conf=args.conf, verbose=False)
            picked = pick_main_person(results[0]) if results else None
            kps = None
            if picked is not None:
                person, conf = picked
                kps = []
                for j in range(17):
                    if j < len(person):
                        x = float(person[j][0])
                        y = float(person[j][1])
                        c = float(conf[j]) if conf is not None and j < len(conf) else 1.0
                        kps.append([round(x, 2), round(y, 2), round(c, 3)])
                    else:
                        kps.append([0.0, 0.0, 0.0])
        if kps is not None:
            frames.append({"t": round(idx / fps, 4), "kps": kps})
        processed += 1
        idx += 1
        if processed % 30 == 0:
            sys.stdout.write("\r  已处理 %d 帧，获得 %d 帧关键点" % (processed, len(frames)))
            sys.stdout.flush()

    cap.release()
    if not frames:
        sys.exit("没有检出任何人体关键点：请确认画面里有完整的人形、光线充足，"
                 "或降低 --conf 阈值（当前 %.2f）重试。" % args.conf)

    out_path = args.out or (os.path.splitext(args.video)[0] + "_pose.json")
    payload = {
        "video": os.path.basename(args.video),
        "fps": round(fps, 2),
        "width": W,
        "height": H,
        "model": model_name,
        "engine": engine,
        "stride": max(1, int(args.stride)),
        "srcFps": round(fps, 2),
        "frames": frames,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print("\n完成：%d 帧关键点 → %s" % (len(frames), out_path))

    if want_overlay:
        out = args.overlay_out or skeleton_path(args.video)
        n = write_overlay(cv2, args.video, payload, out, args.bg,
                          args.codec, not args.no_ids)
        if not n:
            sys.exit("骨架视频未写出任何帧，JSON 已正常生成。")
    print("接下来：打开 coach-workbench.html →「视频评估」→ 导入 JSON；"
          "骨架 MP4 可直接播放复盘。")


if __name__ == "__main__":
    main()
