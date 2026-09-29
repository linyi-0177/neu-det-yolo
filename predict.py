# -*- coding: utf-8 -*-
"""
NEU-DET 推理测试：用训练好的 best.pt 检测图片
在 (cv) 环境下运行：python predict.py
"""

from ultralytics import YOLO
from pathlib import Path

# ── 配置区 ──────────────────────────────────────────
WEIGHTS = r"D:\2026\python project\yolo_train\runs\yolov8n_neu_det\weights\best.pt"

# 要检测的图片来源，两种方式二选一：
SOURCE = r"D:\2026\python project\yolo_train\dataset\NEU-DET\train\images"  # 整个文件夹
# SOURCE = r"某张图片的完整路径.jpg"                                        # 单张图片
# ────────────────────────────────────────────────────

# 类别英文 → 中文对照，打印结果时更好读
CLASS_CN = {
    "crazing": "裂纹",
    "inclusion": "夹杂",
    "patches": "斑块",
    "pitted_surface": "点蚀表面",
    "rolled-in_scale": "氧化铁皮压入",
    "scratches": "划痕",
}


def main():
    model = YOLO(WEIGHTS)

    # stream=True 逐张处理，避免大批量图片占内存
    results = model.predict(
        source=SOURCE,
        conf=0.25,        # 置信度阈值，低于它的框不显示
        save=True,        # 保存画好框的结果图
        device=0,         # 用 GPU
    )

    print("=" * 50)
    total = 0
    for r in results:
        img_name = Path(r.path).name
        boxes = r.boxes
        total += len(boxes)
        if len(boxes) > 0:
            # 注意：boxes.cls 里是 GPU 张量，必须先 int() 转成 Python 整数才能当字典键
            names = [CLASS_CN.get(model.names[int(c)], model.names[int(c)]) + f"({float(b.conf):.2f})"
                     for c, b in zip(boxes.cls, boxes.conf)]
            print(f"{img_name}: 检测到 {len(boxes)} 个缺陷 -> {', '.join(names)}")
    print("=" * 50)
    print(f"共检测出 {total} 个缺陷实例")
    print(f"结果图已保存到: {Path(r.save_dir)}")


if __name__ == "__main__":   # Windows 多进程保护，和 train.py 同理
    main()
