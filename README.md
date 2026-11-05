# 羽毛球专项训练体系 · AthleteIQ

单文件、零依赖、可断网的羽毛球专项训练系统：技术教学 + 体能训练 + 运动康复，等级定级、每日执行清单、视频动作评估与骨架分析视频、负荷监控、瓶颈诊断与 4 周突破计划。

**在线访问**：<https://gxsri.github.io/badminton-coach-workbench/>
**仓库**：<https://github.com/gxsri/badminton-coach-workbench>

本地使用：直接双击 `index.html`，无需服务器、无需 npm、无需联网。所有数据保存在浏览器 `localStorage`。

---

## 三个核心应用

| 文件 | 作用 |
|---|---|
| `index.html` | 导航台：所有应用、文档、快速入口与完整文件索引 |
| `coach-workbench.html` | 专项教练工作台：9 个模块（今日 / 等级定级 / 内容库 / 训练记录 / 视频评估 / 数据接入 / 瓶颈诊断 / **规则配置** / 档案），专业与简单双模式 |
| `data-collector.html` | 离线多人数据采集器：打卡、训练日志、周期问卷、穿戴 CSV、体能测评、视频记录，导出工作台可直接合并的采集包 |
| `data-schema.html` | 数据字典、采集频率 SOP、质量控制、多人命名与泛化验收清单 |

## 泛化能力（不是演示页面）

系统结论全部由「真实数据 + 可编辑规则」重算：

- **配置层**：技术库、体能动作库、技术配套动作、预康复处方、测评基准、周期模板、等级表、定级问卷、每日/周期问卷、姿态指标阈值、短板封顶规则、文案模板，以及 70+ 个算法参数（准备度扣分、体能权重、ACWR 阈值、降载系数、4 周突破系数、定级映射、预警阈值等）全部存于 `DB.cfg`，在「规则配置」页可视化编辑或 JSON 编辑，支持配置包导入导出。
- **空数据诚实**：没有定级 / 日志 / 打卡 / 测评时，对应领域显示「未评估」并列出缺口，不再用默认分数冒充真实水平。诊断页会明确写「数据不足，暂不给出确定性结论」。
- **A/B 灵敏度验证**：「规则配置」页内置两组虚拟运动员，只改 6 个输入值，运行同一套算法并对比 13 项输出（准备度、ACWR、三域得分、主要瓶颈、清单、周负荷目标、预警数等），可直接确认「输入不同 → 输出不同」。
- **采集闭环**：采集器按配置包动态生成表单 → 导出 `bwb.collect` / `bwb.collect.batch` → 工作台按 ID / 日期去重合并并留痕 → 清单、等级、诊断、计划自动重算。
- **演示数据可辨识**：生成的演示记录带 `demo:true` 与 `[演示]` 标记，可一键清除，不覆盖真实数据。

## 数据接入与视频评估

- 训练日志（session-RPE）、每日问卷、周期问卷、穿戴 CSV（date / resting_hr / hrv / sleep_hours / steps / load）、体能测评、备份 JSON 导入导出。
- 视频评估双引擎：内置 Canvas 逐帧像素运动分析（无需任何依赖）；或导入 COCO-17 关键点 JSON 做姿态解算（肩外展、肘伸展、击球点高度、躯干旋转、髋肩分离、膝内扣、腕速、随挥、落地稳定性等）。
- 真实姿态解算用随项目提供的 `yolo-pose-bridge.py`。**默认走 ONNX + OpenCV DNN**，只需 `opencv-python` 和 `numpy`，不需要 torch / ultralytics，也不需要联网下载权重（模型 `yolo11n-pose.onnx` 随仓库提供）：

  ```bash
  pip install opencv-python numpy
  python yolo-pose-bridge.py 你的视频.mp4 -o 输出.pose.json --overlay
  ```

  然后在工作台「视频评估」导入该 JSON。不装依赖也能用内置像素分析或示例姿态数据。模型文件丢失时执行 `python yolo-pose-bridge.py --fetch-model` 重新下载；也可用 `--engine ultralytics --model yolov8n-pose.pt` 切回 ultralytics 引擎。实测 H.265/HEVC 的微信导出视频可直接解码。

## 骨架分析视频

视频评估不只给分数，还会把骨架画回画面，产出可播放、可导出的分析视频。两条路径按需选择：

| 路径 | 产出 | 适用 | 依赖 |
|---|---|---|---|
| 工作台「视频评估 → 骨架分析视频」 | WebM（浏览器录制）、当前帧 PNG | 单条视频快速复盘、随堂讲解 | 现代浏览器即可，无需 Python |
| `yolo-pose-bridge.py --overlay` | MP4（逐帧写入，可批量） | 批量、高清、需要 MP4 归档 | Python + `opencv-python` |

浏览器端流程：上传视频 → 导入 YOLO 姿态 JSON（或点「载入示例骨架（无需视频）」/「用内置分析生成轨迹视频」）→ 骨架分析视频卡片显示 COCO-17 骨架逐帧叠加。可选「叠加原视频」核对贴合度或「纯骨架」看动作链路，支持 0.5×–2× 回放、时间轴拖动、片段起止秒、运动轨迹 / 关键点 / 编号 / 数据面板开关，然后「导出骨架分析视频（WebM）」或「保存当前帧 PNG」。未做姿态解算时，内置像素分析会输出重心与运动能量轨迹视频。

导出文件名：`bwb-skeleton-<技术>-<日期>.webm`、`bwb-skeleton-frame-<日期>.png`。WebM 采用 `canvas.captureStream(30)` + `MediaRecorder` 按真实时间录制，因此需要让页面保持在前台直到录制完成；Chrome / Edge / Firefox 支持，Safari 等不支持 `MediaRecorder` 的浏览器可改用本地 MP4 路径。

本地脚本用法（默认 `onnx` 引擎，模型随仓库提供）：

```bash
pip install opencv-python numpy

# 模式 A：解算 JSON，并同时产出骨架 MP4
python yolo-pose-bridge.py 你的视频.mp4 -o 输出.pose.json --overlay

# 模式 B：已有姿态 JSON，跳过模型直接重出骨架视频（可换背景 / 关编号）
python yolo-pose-bridge.py 你的视频.mp4 --pose-json 输出.pose.json --bg plain --no-ids --overlay-out 骨架复盘.mp4

# 模型丢失时重新获取；或切换到 ultralytics 引擎（需 torch）
python yolo-pose-bridge.py --fetch-model
python yolo-pose-bridge.py 你的视频.mp4 --engine ultralytics --model yolov8n-pose.pt --overlay
```

关键参数：`--overlay` 额外生成 MP4；`--overlay-out` 指定 MP4 路径（隐含 `--overlay`）；`--bg video|plain` 叠加原视频或浅色纯骨架背景；`--pose-json` 从已有 JSON 生成骨架视频、跳过姿态推理；`--no-ids` 关闭关键点编号；`--codec` 指定 fourcc（默认 `mp4v`）；`--stride N` 每 N 帧解算一帧（输出帧率自动按抽样比调整）；`--conf` 置信阈值（默认 0.35）；`--engine auto|onnx|ultralytics`。默认输出 `<视频名>_pose.json` 与 `<视频名>_skeleton.mp4`。JSON 顶层含 `video`、`fps`、`width`、`height`、`model`、`engine`、`stride`、`srcFps`、`frames:[{t,kps}]`；`width/height` 供工作台把骨架精确贴回原画面，`stride` 让 MP4 回放保持真实时长。

**编码兼容**：浏览器端 `<video>` 依赖浏览器自身的解码能力，H.265/HEVC 的微信视频在部分浏览器里无法播放。本地脚本用 OpenCV 顺序解码，H.264 与 H.265 都能处理；若浏览器不支持，直接用上面的 MP4 路径，或先把视频转成 H.264。

## 等级体系

定级问卷参考中羽在线 0–7 级分级讨论帖的俗称与能力描述（羽外高人 → 呼风唤羽），采用 10 维加权均分 + 短板封顶 + 竞技成绩校正；结果自动写入技术基线与目标等级。等级表、题目、权重、封顶规则、映射公式均可在配置页修改。仅作自评参考，不替代教练现场评估。

## 目录

静态资料：`coach-guide.html`（13 章教练指导书）、`coach-manual-v2.html`（7-16 岁速查手册）、`level-assessment.html`（青少儿交互评估）、`kids-training-program-part{1,2,3}.html`（儿童训练方案）、`drill-cards.html`（A4 训练卡片）、`poster-cheatsheet.html`、`tracking-and-parent-letter.html`、`test-report.html`、`coach-manual-7to16.html`（归档）。

## 验证记录

- 工作台核心回归：55 用例（9 视图 / 空数据诚实性 / 配置层与参数 / 定级差异 / A-B 泛化 / 采集包去重 / 演示数据 / 视频双引擎 / 负荷与预警）—— 0 失败
- 数据采集器：23 用例 —— 0 失败
- 数据规范页：3 用例 —— 0 失败
- 骨架分析视频引擎：65 用例（姿态 JSON 归一化 / COCO-17 连线 / 叠加与纯骨架绘制 / 片段裁剪 / 回放与录制状态 / 导出）—— 0 失败
- 真实姿态 JSON 消费链路：48 用例（归一化 / 包围盒 / 坐标变换 / 姿态评分 / 骨架卡片渲染）—— 0 失败
- `yolo-pose-bridge.py`：`py_compile`、`--help`、`BONES` 配色、`draw_skeleton()`、`write_overlay()`、`--pose-json`、ONNX 引擎分发、坐标还原、抽样帧率 —— 0 失败
- **真实训练视频端到端验证**：3 段 H.265/HEVC 微信视频（1280×720 / 720×1280 / 1280×720，30fps，85 / 138 / 254 帧）全部产出骨架 MP4 与姿态 JSON；477/477 帧检出人体，每帧可见关键点最少 13、平均 16.2–17.0；三个 MP4 均可解码且帧数一致。产出的真实姿态 JSON 在工作台归一化、姿态评分、骨架卡片渲染链路 48 用例 —— 0 失败
- 全站静态审计：16 个 HTML、167 个链接与锚点全部可达，50 + 25 个点击型动作处理器全覆盖，外部 CSS/JS/图片依赖 0，所有内联脚本 `node --check` 通过
- 浏览器端 WebM 导出依赖 `MediaRecorder` / `canvas.captureStream()`，需在真实浏览器中验证；不支持的浏览器请改用 `--overlay` 生成 MP4

## 隐私与免责

所有记录仅存于本机浏览器，不上传服务器；换设备用 JSON 导入导出迁移。力量、灵敏度、关节角度阈值均为通用运动科学参考值，不替代医疗诊断。出现夜间痛、静息痛、关节绞锁或不稳、明显肿胀、放射性疼痛时应停止训练并寻求专业医疗评估。
