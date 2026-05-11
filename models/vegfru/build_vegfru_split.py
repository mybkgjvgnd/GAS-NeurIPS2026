import os
from tqdm import tqdm

BASE_DIR = "/mnt/ljh-21/organized_data_simple/vegfru"
LIST_DIR = os.path.join(BASE_DIR, "vegfru_list")
OUTPUT_DIR = os.path.join(BASE_DIR, "vegfru_split")

SPLITS = ["train", "val", "test"]

for split in SPLITS:
    txt_path = os.path.join(LIST_DIR, f"vegfru_{split}.txt")
    split_dir = os.path.join(OUTPUT_DIR, split)

    os.makedirs(split_dir, exist_ok=True)

    with open(txt_path, "r") as f:
        lines = f.readlines()

    print(f"Processing {split}...")

    for line in tqdm(lines):
        rel_path, label = line.strip().split()
        src_path = os.path.join(BASE_DIR, rel_path)

        # 类别名来自文件夹名
        class_name = rel_path.split("/")[1]

        class_dir = os.path.join(split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        filename = os.path.basename(rel_path)
        dst_path = os.path.join(class_dir, filename)

        if not os.path.exists(dst_path):
            os.symlink(src_path, dst_path)

print("✅ Done building vegfru_split")