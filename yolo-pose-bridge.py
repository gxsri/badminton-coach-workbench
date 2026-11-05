#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
yolo-pose-bridge.py —— 羽毛球动作视频的本地姿态解算桥接脚本

用途：把一段训练视频解算成 COCO-17 关键点序列 JSON，
      供 coach-workbench.html 的「YOLO 姿态解算结果导入」读取，
      从而得到真实关节角度级别的技术评估（而非像素级粗略估计）。

安装依赖（只需一次）：
    pip install ultralytics opencv-python

基本用法：
    python yolo-pose-bridge.py smash.mp4
    python yolo-pose-bridge.py smash.mp4 -o smash_pose.json
    python yolo-pose-bridge.py smash.mp4 --model yolov8s-pose.pt --stride 1 --max-frames 300

参数说明：
    video             输入视频文件（mp4 / mov / avi 等，OpenCV 能读即可）
    -o/--out          输出 JSON 路径（默认：与视频同名 + _pose.json）
    --model           YOLO 姿态模型，默认 yolov8n-pose.pt（n=最快，s/m 更准但更慢）
    --stride          每隔 N 帧处理一帧（默认 1，即逐帧；视频很长时可设 2-3）
    --max-frames      最多处理多少帧（默认 600，约 20 秒 @30fps）
    --conf            检测置信度阈值（默认 0.5）
    --device          推理设备：cpu / 0（GPU）等，默认自动

输出格式（与本工作台约定一致）：
    {
      "video": "smash.mp4",
      "fps": 30.0,
      "model": "yolov8n-pose.pt",
      "frames": [
        {"t": 0.0,  "kps": [[x, y, conf], ... 17 个点 ...]},
        {"t": 0.033,"kps": [[x, y, conf], ...]}
      ]
    }

COCO-17 关键点顺序：
    0 nose  1 L-Eye  2 R-Eye  3 L-Ear  4 R-Ear
    5 L-Shoulder  6 R-Shoulder  7 L-Elbow  8 R-Elbow  9 L-Wrist  10 R-Wrist
    11 L-Hip  12 R-Hip  13 L-Knee  14 R-Knee  15 L-Ankle  16 R-Ankle

拍摄建议：侧面或斜侧后 45°，全身入镜，机位固定，包含 3 次以上完整动作。

注意：全部计算在本机完成，视频不会上传到任何服务器。
"""

import argparse
import json
import os
import sys

def parse_args():
    ap = argparse.ArgumentParser(description="YOLOv8-pose 视频 → COCO-17 关键点 JSON")
    ap.add_argument("video", help="输入视频文件路径")
    ap.add_argument("-o", "--out", default=None, help="输出 JSON 路径")
    ap.add_argument("--model", default="yolov8n-pose.pt", help="姿态模型权重")
    ap.add_argument("--stride", type=int, default=1, help="每隔 N 帧处理一帧")
    ap.add_argument("--max-frames", type=int, default=600, help="最多处理帧数")
    ap.add_argument("--conf", type=float, default=0.5, help="检测置信度阈值")
    ap.add_argument("--device", default=None, help="推理设备，如 cpu / 0")
    return ap.parse_args()


def open_video(path):
    try:
        import cv2
    except ImportError:
        sys.exit("缺少依赖 opencv-python，请先执行：pip install opencv-python")
    return cv2.VideoCapture(path)


def load_model(model_path, device):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "缺少依赖 ultralytics，请先执行：\n"
            "    pip install ultralytics\n"
            "（首次运行会自动下载模型权重，需要联网一次）"
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


def main():
    args = parse_args()
    if not os.path.isfile(args.video):
        sys.exit("找不到视频文件：%s" % args.video)

    cap = open_video(args.video)
    if not cap.isOpened():
        sys.exit("无法打开视频（编码不支持或文件损坏）：%s" % args.video)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    model = load_model(args.model, args.device)

    frames, idx, processed = [], 0, 0
    print("正在解算：%s（fps=%.1f，总帧数=%d）" % (os.path.basename(args.video), fps, total))

    while processed < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.stride != 0:
            idx += 1
            continue
        results = model.predict(frame, conf=args.conf, verbose=False,
                                device=args.device) if args.device else \
                  model.predict(frame, conf=args.conf, verbose=False)
        picked = pick_main_person(results[0]) if results else None
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
            frames.append({"t": round(idx / fps, 4), "kps": kps})
        processed += 1
        idx += 1
        if processed % 30 == 0:
            sys.stdout.write("\r  已处理 %d 帧，获得 %d 帧关键点" % (processed, len(frames)))
            sys.stdout.flush()

    cap.release()
    if not frames:
        sys.exit("没有检出任何人体关键点：请确认画面里有完整的人形、光线充足，或降低 --conf 阈值重试。")

    out_path = args.out or (os.path.splitext(args.video)[0] + "_pose.json")
    payload = {
        "video": os.path.basename(args.video),
        "fps": round(fps, 2),
        "model": os.path.basename(args.model),
        "frames": frames,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print("\n完成：%d 帧关键点 → %s" % (len(frames), out_path))
    print("接下来：打开 coach-workbench.html →「视频评估」→ 选择对应技术 → 导入该 JSON。")


if __name__ == "__main__":
    main()
