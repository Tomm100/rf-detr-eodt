#!/bin/bash
#SBATCH --job-name=eodt_benchmark
#SBATCH --account=IscrC_EoDT
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/leonardo_work/IscrC_EoDT/tcentonz/runs/slurm/%x-%j.out
#
# Latency benchmark of EODT_SPEC.md §6 (step 13). It measures every model on one A100 with the protocol of
# configs/benchmark/a100.yaml. The node is reserved (--exclusive), so no other job shares it. Each model runs in its
# own process, one after the other: EoDT, the rings of the baseline and RF-DETR's published checkpoints.
# From the code folder on the login node, after prepare_offline.sh:
#
#   sbatch scripts/leonardo/benchmark.sh
#
# Latency does not depend on the weights. CHECKPOINT is the name of a trained checkpoint inside the output_dir of every
# experiment; if it is set, every trained model loads it (it must exist for all of them):
#
#   sbatch --export=ALL,CHECKPOINT=last_ema.pth scripts/leonardo/benchmark.sh
#
# The records go to runs/benchmark/<job id>/<model>.json. The test of step 13 runs the job twice and compares the
# two jobs (same parameters and MACs, mean latencies within 2%):
#
#   python scripts/eodt/benchmark.py compare /leonardo_work/IscrC_EoDT/tcentonz/runs/benchmark/<job 1> \
#       /leonardo_work/IscrC_EoDT/tcentonz/runs/benchmark/<job 2>
#
# Time limit: 9 models at about 3 minutes each on an A100 (loading, compilation, 500 + 5000 forward passes).
# This is estimated from the local trial, see the summary of step 13. The rest is margin.
set -eo pipefail
source /leonardo_work/IscrC_EoDT/tcentonz/rf-detr-eodt/scripts/leonardo/env.sh

PROTOCOL=configs/benchmark/a100.yaml
OUT=$WORK_DIR/runs/benchmark/$SLURM_JOB_ID
CHECKPOINT=${CHECKPOINT:-}

MODELS="
configs/eodt/eodt_vits_576_l2-3_selection.yaml
configs/eodt/eodt_vits_576_l2-3_learned.yaml
configs/baseline/baseline_medium_win.yaml
configs/baseline/baseline_medium_nowin.yaml
configs/baseline/baseline_medium_nowin_noproj.yaml
"
PUBLISHED="nano small medium large"

nvidia-smi
for yaml in $MODELS; do
    name=$(basename "$yaml" .yaml)
    weights=()
    if [ -n "$CHECKPOINT" ]; then
        weights=(--checkpoint "$WORK_DIR/runs/$name/$CHECKPOINT")
    fi
    srun python scripts/eodt/benchmark.py run $PROTOCOL --model "$yaml" "${weights[@]}" --output "$OUT/$name.json"
done
for size in $PUBLISHED; do
    srun python scripts/eodt/benchmark.py run $PROTOCOL --published "$size" --output "$OUT/rfdetr_$size.json"
done
echo "records in $OUT"
