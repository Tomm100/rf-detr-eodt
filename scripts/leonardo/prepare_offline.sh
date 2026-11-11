#!/bin/bash
# Prepare Leonardo for the offline jobs of EoDT (EODT_SPEC.md §9, step 6, point 4). Run it ON THE LOGIN NODE, which
# has internet, after every change of backbone or dataset:
#
#   bash scripts/leonardo/prepare_offline.sh
#
# It downloads the DINOv2 weights of the three sizes into HF_HOME, for EoDT (timm) and for the baseline (Hugging Face
# facebook/dinov2-with-registers-*, read by RF-DETR's loader, EODT_SPEC.md §8 point 4), and DINOv2-B without registers
# for the upper bound of Table 7 (facebook/dinov2-base, §10), checks that they load without internet as on the
# compute nodes, downloads RF-DETR's published checkpoints of the latency benchmark into RF_HOME and checks their MD5,
# checks that COCO and coco_mini are in place and creates the folder of the SLURM logs.
set -eo pipefail
source /leonardo_work/IscrC_EoDT/tcentonz/rf-detr-eodt/scripts/leonardo/env.sh

mkdir -p "$WORK_DIR/runs/slurm"

SIZES="small base large"
for offline in 0 1; do  # download, then load from the cache only, as a compute node does
    HF_HUB_OFFLINE=$offline python - $SIZES <<'EOF'
import os
import sys

import timm

from rfdetr.models.eodt import load_native_dinov2

state = "loaded offline" if os.environ["HF_HUB_OFFLINE"] == "1" else "downloaded"
for size in sys.argv[1:]:
    name = f"vit_{size}_patch14_reg4_dinov2"
    timm.create_model(name, pretrained=True, num_classes=0)  # EoDT
    print(f"{name}: {state}")
    load_native_dinov2(size)  # baseline: RF-DETR's loader at patch 14, which also checks every key
    print(f"facebook/dinov2-with-registers-{size}: {state}")
load_native_dinov2("base", registers=False)  # upper bound of Table 7: DINOv2-B without registers
print(f"facebook/dinov2-base: {state}")
EOF
done

# RF-DETR's published checkpoints for the latency benchmark (EODT_SPEC.md §6) into RF_HOME (env.sh), with their MD5
mkdir -p "$RF_HOME"
python - <<'EOF'
import hashlib
import os

from rfdetr.assets.model_weights import ModelWeights, download_pretrain_weights
from rfdetr.models.eodt.benchmark import PUBLISHED, published_config

for name in PUBLISHED:
    path = published_config(name).pretrain_weights
    download_pretrain_weights(path)  # nothing to do if the file is already there
    with open(path, "rb") as file:
        md5 = hashlib.file_digest(file, "md5").hexdigest()
    if md5 != ModelWeights.from_filename(os.path.basename(path)).md5_hash:
        raise SystemExit(f"{path}: wrong MD5, delete the file and run this script again")
    print(f"RF-DETR {name}: {path}")
EOF

# image counts of COCO 2017 (§6) and of coco_mini (CLAUDE.md)
python - <<'EOF'
from pathlib import Path

datasets = {
    "/leonardo_scratch/fast/IscrC_EoDT/coco": {"train": 118287, "val": 5000},
    "/leonardo_work/IscrC_EoDT/tcentonz/data/coco_mini": {"train": 20, "val": 20},
}
for root, splits in datasets.items():
    for split, expected in splits.items():
        annotations = (Path(root) / "annotations" / f"instances_{split}2017.json").exists()
        images = len(list((Path(root) / f"{split}2017").glob("*.jpg")))
        print(f"{root} {split}: {images} images (expected {expected}), annotations {'found' if annotations else 'MISSING'}")
        if images != expected or not annotations:
            raise SystemExit(f"{root}: the {split} split is not complete")
EOF
echo "Leonardo is ready for the offline jobs"
