"""
NEU-DET 原始标签可视化（简化版，命令行友好）
- 无窗口弹窗，只保存图
- 每类抽 1 张 + 再随机抽 4 张，共 10 张
- 输出到 D:\2026\python project\yolo_train\labels_view\
"""
import random
from pathlib import Path
import cv2

IMG_DIR = Path(r"D:\2026\python project\yolo_train\dataset\NEU-DET\train\images")
LBL_DIR = Path(r"D:\2026\python project\yolo_train\dataset\NEU-DET\train\labels")
OUT_DIR = Path(r"D:\2026\python project\yolo_train\labels_view")
OUT_DIR.mkdir(exist_ok=True)

COLORS = {
    0: (0,   0,   255), 1: (0,   255, 0), 2: (255, 0,   0),
    3: (0,   255, 255), 4: (255, 0,   255), 5: (255, 255, 0),
}
NAMES = {0: "crazing", 1: "inclusion", 2: "patches",
         3: "pitted_surface", 4: "rolled-in_scale", 5: "scratches"}


def view_one(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    h, w = img.shape[:2]
    lbl = LBL_DIR / (img_path.stem + ".txt")
    n = 0
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            if not line.strip():
                continue
            try:
                cid, cx, cy, bw, bh = map(float, line.split())
            except ValueError:
                continue
            cid = int(cid)
            x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS.get(cid, (255, 255, 255)), 1)
            cv2.putText(img, NAMES.get(cid, str(cid)), (x1, max(y1 - 3, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS.get(cid, (255, 255, 255)), 1)
            n += 1
    cv2.imwrite(str(OUT_DIR / img_path.name), img)
    return n


if __name__ == "__main__":
    all_imgs = list(IMG_DIR.glob("*.jpg"))
    print(f"数据集共 {len(all_imgs)} 张图")

    samples = []
    for prefix in ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]:
        cands = [p for p in all_imgs if p.name.startswith(prefix)]
        if cands:
            samples.append(random.choice(cands))

    rest = [p for p in all_imgs if p not in samples]
    samples.extend(random.sample(rest, min(4, len(rest))))

    total = 0
    for p in samples:
        c = view_one(p)
        total += c
        print(f"  {p.name}: {c} 个标注框")

    print(f"\n完成，共画 {total} 个标注框")
    print(f"结果目录: {OUT_DIR}")