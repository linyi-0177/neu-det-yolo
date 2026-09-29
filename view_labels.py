"""
NEU-DET 原始标签可视化（简化版，命令行友好）
- 无窗口弹窗，只保存图
- 每类抽 1 张 + 再随机抽 4 张
- 输出到项目根目录下的 labels_view/

用途有两层：
1. 直观查看标注质量（框是否贴合缺陷区域）
2. 间接校验标注完整性 —— 标注文件缺失的图片画出来会是空白，一眼能看出问题
"""
import random
from pathlib import Path

import cv2

# 项目根目录 = 本文件所在目录，换电脑 / 换盘符无需改代码
ROOT = Path(__file__).resolve().parent

IMG_DIR = ROOT / "dataset" / "NEU-DET" / "train" / "images"
LBL_DIR = ROOT / "dataset" / "NEU-DET" / "train" / "labels"
OUT_DIR = ROOT / "labels_view"
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
        # 框数为 0 说明该图没有标注文件（或标注为空），是数据问题的信号
        flag = "  <-- 无标注，建议检查" if c == 0 else ""
        print(f"  {p.name}: {c} 个标注框{flag}")

    print(f"\n完成，共画 {total} 个标注框")
    print(f"结果目录: {OUT_DIR}")
