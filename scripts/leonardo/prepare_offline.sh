#!/bin/bash
# Prepare Leonardo for the offline jobs of EoDT (EODT_SPEC.md §9, step 6, point 4). Run it ON THE LOGIN NODE, which
# has internet, after every change of backbone or dataset:
#
#   bash scripts/leonardo/prepare_offline.sh
#
# It downloads the DINOv2 weights of the three sizes into HF_HOME, checks that they load without internet as on the
# compute nodes, checks that COCO and coco_mini are in place and creates the folder of the SLURM logs.
set -eo pipefail
source /leonardo_work/IscrC_EoDT/tcentonz/rf-detr-eodt/scripts/leonardo/env.sh

mkdir -p "$WORK_DIR/runs/slurm"

BACKBONES="vit_small_patch14_reg4_dinov2 vit_base_patch14_reg4_dinov2 vit_large_patch14_reg4_dinov2"
for offline in 0 1; do  # download, then load from the cache only, as a compute node does
    HF_HUB_OFFLINE=$offline python - $BACKBONES <<'EOF'
import os
import sys

import timm

for name in sys.argv[1:]:
    timm.create_model(name, pretrained=True, num_classes=0)
    print(f"{name}: {'loaded offline' if os.environ['HF_HUB_OFFLINE'] == '1' else 'downloaded'}")
EOF
done

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
