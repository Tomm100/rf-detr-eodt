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
# Overfit of EoDT on the 20 images of coco_mini, in bf16 on one A100: the criteria of step 5 again, as the first
# job of step 6 (EODT_SPEC.md §9). From the code folder on the login node:
#
#   sbatch scripts/leonardo/overfit.sh
#
# Normal queue, not the debug QOS: 2400 steps take an estimated 11-28 minutes on an A100 (66 on the GTX 1650),
# too close to its 30-minute limit; the uncertain part is last.ckpt (344 MB) written to $WORK at every epoch.
# The verdict is at the end of the SLURM output; curves.png and predictions.png are in the output_dir of the YAML.
# W&B (tag "test"): an overfit does not resume, so wandb picks a new run id at every job. Sync from the login node:
#
#   wandb sync /leonardo_work/IscrC_EoDT/tcentonz/runs/eodt_vits_576_l2-3_learned_overfit_leonardo/wandb/offline-run-*
set -eo pipefail
source /leonardo_work/IscrC_EoDT/tcentonz/rf-detr-eodt/scripts/leonardo/env.sh

srun python scripts/eodt/overfit.py configs/eodt/eodt_vits_576_l2-3_learned_overfit_leonardo.yaml
