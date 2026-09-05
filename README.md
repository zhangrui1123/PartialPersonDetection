# PartialPersonDetection

灰度 **640×480** 视频里判断画面是否有人（含边缘截断的半身/局部人体），输出 **有人 / 没人**。

## 模型架构 `configs/yolov8n-gray.yaml`

相对标准 YOLOv8n 的改动：

| 项 | 标准 YOLOv8n | 本仓库 |
|----|----------------|--------|
| 输入 | RGB 3 通道，常按 640×640 letterbox | **1 通道灰度**，原生 **640×480**（`imgsz: [480, 640]` + `rect`） |
| 检测头 | P3 / P4 / P5（stride 8/16/32） | **P2 / P3 / P4 / P5**（stride **4**/8/16/32），利于小目标和截断人体 |
| 类别 | COCO 80 类 | 单类 `person` |

640×480 可被 stride 32 整除，特征图为：

```
输入  1 × 480 × 640
  │  Conv  stride 2     P1/2   16 × 240 × 320
  │  Conv  stride 2     P2/4   32 × 120 × 160  ──┐
  │  C2f + Conv /8      P3/8   64 ×  60 ×  80  ──┤
  │  C2f + Conv /16     P4/16 128 ×  30 ×  40  ──┤
  │  C2f + Conv /32     P5/32 256 ×  15 ×  20  ──┤
  │  SPPF                                            │
  └──────── FPN/PAN 上采样 + 拼接 ─────────────────┘
                    Detect(P2, P3, P4, P5) → person 框
```

第一层从 RGB 预训练 `weights/yolov8n.pt` 把 3 通道卷积按通道平均迁到 1 通道；P2 头随机初始化后一起训练。不要用 P6（480 不能被 64 整除）。

## 训推流程

```
data/raw/coco          原始 COCO 2017
       │
       ▼  prepare_data.py
data/gray              640×480 灰度缓存
       │  + 截断人体增强
       ▼
data/person            训练/验证集（person.yaml，channels: 1）
       │
       ▼  train.py  ← configs/train.yaml
runs/yolov8n_gray/     训练日志
weights/best.pt        发布权重（可将 best 拷到此处）
       │
       ▼  infer.py --source 图/目录/视频
有人 / 没人            默认 conf=0.03
runs/predict/occupancy/
```

1. **数据** `python3 prepare_data.py`  
   把 COCO 转到 640×480 灰度，只保留 `person` 框，并生成截断半身样本，写入 `data/person`。  
   已有灰度缓存时：`python3 prepare_data.py --skip-convert`。  
   冒烟：`python3 prepare_data.py --debug` → `data/person_debug`。

2. **训练** `python3 train.py`  
   读 `configs/train.yaml`：架构 `yolov8n-gray.yaml`，50 epoch，batch 128，输入 `[480, 640]`。  
   权重默认写到 `runs/yolov8n_gray/weights/best.pt`。发布用：拷到 `weights/best.pt`。

3. **推理** `python3 infer.py --source your.mp4`  
   默认 `weights/best.pt`，`imgsz [480, 640]`，`--conf 0.03`（精确率/召回较均衡）。  
   可视化与 JSON：`runs/predict/occupancy/`。

一键：

```bash
python3 -m pip install -r requirements.txt
python3 run_pipeline.py                  # 数据 → 训练 → 验证集推理
python3 infer.py --source your.mp4
```

已有数据和权重时：

```bash
python3 train.py --epochs 50 --device 0
python3 infer.py --weights weights/best.pt --conf 0.03 --source your.mp4
```
