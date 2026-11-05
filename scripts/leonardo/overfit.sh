#!/bin/bash
#SBATCH --job-name=eodt_overfit
#SBATCH --account=IscrC_EoDT
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/leonardo_work/IscrC_EoDT/tcentonz/runs/slurm/%x-%j.out
#
# Overfit of EoDT or of the baseline on the 20 images of coco_mini, in bf16 on one A100: the criteria of step 5 again,
# to verify the infrastructure of step 6 on the two models of step 10 (EODT_SPEC.md §9). The YAML file is the
# argument, the job is named after it; from the code folder on the login node:
#
#   CONFIG=configs/eodt/eodt_vits_576_l2-3_selection_overfit_leonardo.yaml
#   sbatch -J $(basename $CONFIG .yaml) scripts/leonardo/overfit.sh $CONFIG
#
# and the same with CONFIG=configs/baseline/baseline_medium_win_overfit_leonardo.yaml for the baseline.
# The arguments are those of scripts/eodt/overfit.py. Normal queue, not the debug QOS: 2400 steps take an estimated
# 11-28 minutes on an A100 (65 for EoDT selection and 60 for the baseline on the GTX 1650), too close to its 30-minute
# limit; the uncertain part is last.ckpt written to $WORK at every epoch.
# The verdict is at the end of the SLURM output; curves.png and predictions.png are in the output_dir of the YAML.
# W&B (tag "test"): an overfit does not resume, so wandb picks a new run id at every job. Sync from the login node:
#
#   wandb sync <output_dir of the YAML>/wandb/offline-run-*
set -eo pipefail
source /leonardo_work/IscrC_EoDT/tcentonz/rf-detr-eodt/scripts/leonardo/env.sh

# srun does not inherit --cpus-per-task from sbatch: pass it, so that the process has its 8 cores
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" python scripts/eodt/overfit.py "$@"
