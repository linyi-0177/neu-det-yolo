# -*- coding: utf-8 -*-
"""
NEU-DET 钢材表面缺陷检测 —— YOLOv8 训练脚本
数据集：6 类（crazing / inclusion / patches / pitted_surface / rolled-in_scale / scratches）
模型  ：yolov8n（最小最轻，RTX 4060 8GB 完全无压力）

注意：Windows 下 DataLoader 多进程必须把代码包在 main() 里 + __main__ 保护，
否则会报 "An attempt has been made to start a new process before the
current process has finished its bootstrapping phase" 错误。
"""
from ultralytics import YOLO


def main():
    # 1. 加载预训练模型（首次会自动下载 yolov8n.pt 约 6MB）
    model = YOLO("yolov8n.pt")

    # 2. 启动训练
    results = model.train(
        data=r"D:\2026\python project\yolo_train\data.yaml",   # 数据配置
        epochs=50,                   # 训练轮数
        imgsz=640,                   # 输入图像尺寸（原图 200x200 会被上采样到 640）
        batch=16,                    # 批大小（8GB 显存能跑，4060 还能上 32）
        device=0,                    # 0 = 用第一块 GPU
        workers=4,                   # 数据加载线程数
        project=r"D:\2026\python project\yolo_train\runs",  # 训练结果根目录
        name="yolov8n_neu_det",     # 本次训练的子目录名
        exist_ok=True,               # 同名目录允许覆盖
        patience=10,                 # 早停：val 上 10 轮没提升就停
        plots=True,                  # 自动生成 loss/mAP 曲线图
        save=True,                   # 保存 best.pt / last.pt
        amp=False,                   # 跳过 AMP 兼容性检查（避免再下 yolo26n.pt）
    )

    print("=" * 50)
    print("训练完成！")
    print("最佳权重路径：", results.save_dir / "weights" / "best.pt")
    print("=" * 50)


# Windows 多进程保护：子进程 import 本文件时不会执行 main()
if __name__ == "__main__":
    main()
