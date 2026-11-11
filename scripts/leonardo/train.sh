#!/bin/bash
#SBATCH --job-name=eodt
#SBATCH --account=IscrC_EoDT
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo_work/IscrC_EoDT/tcentonz/runs/slurm/%x-%j.out
#
# Train EoDT or the baseline from its YAML file on one node with 4 A100 (EODT_SPEC.md §9, steps 6 and 10): one process
# per GPU, launched by srun, which Lightning recognises as a SLURM DDP run. The job is named after the YAML file.
# From the code folder on the login node:
#
#   CONFIG=configs/eodt/eodt_vits_576_l2-3_selection.yaml
#   sbatch -J $(basename $CONFIG .yaml) scripts/leonardo/train.sh $CONFIG
#
# and the same with CONFIG=configs/baseline/baseline_medium_win.yaml for the baseline of step 10.
# The arguments are those of scripts/leonardo/train.py. A run longer than 24 hours continues in a new job, from
# last.ckpt and in the same W&B run (WANDB_RUN_ID is the name of the YAML file):
#
#   sbatch -J $(basename $CONFIG .yaml) --dependency=afterany:<job id> scripts/leonardo/train.sh $CONFIG --resume
#
# Test job of each model of step 10 (verification of step 6), twice in the debug QOS (30 minutes each): the second
# job resumes the first.
#
#   CONFIG=configs/eodt/eodt_vits_576_l2-3_selection_testjob.yaml
#   sbatch -J $(basename $CONFIG .yaml) --qos=boost_qos_dbg --time=00:30:00 scripts/leonardo/train.sh $CONFIG \
#       --run-type test --limit-train-batches 300 --limit-val-batches 40 --stop-after-epoch 1
#   sbatch -J $(basename $CONFIG .yaml) --qos=boost_qos_dbg --time=00:30:00 --dependency=afterok:<job id> \
#       scripts/leonardo/train.sh $CONFIG --run-type test --limit-train-batches 300 --limit-val-batches 40 --resume
#
# and the same with CONFIG=configs/baseline/baseline_medium_win_testjob.yaml for the baseline.
#
# Upper bound of Table 7 (step 11 (3)): first the two test jobs above with
# CONFIG=configs/baseline/rfdetr_table7_2xl_upperbound_testjob.yaml and with
# CONFIG=configs/baseline/rfdetr_table7_max_upperbound_testjob.yaml. From the first test job of each model take t, the
# median of train/batch_time_s (W&B or metrics.csv), and the epochs K of the run that fit in one job of 24 hours:
#
#   h = 7400 * t * 1.1 / 3600    hours per epoch, +10% for the validation on val2017
#   K = floor(23 / h)            one hour left for start-up, checkpoints and margin
#
# 7400 is the number of steps per epoch: 118287 images of train2017 / 16, the total batch (batch_size 4 per GPU x 4
# GPUs); recompute it if batch or GPUs change. If K = 0 (one epoch longer than 23 hours), do not launch: review the
# plan.
#
# Then the run, as a chain of jobs that each end after a whole epoch: --stop-after-epoch ends the job after that epoch
# with exit code 0, and the next job resumes from last.ckpt only if the previous one succeeded (afterok). EPOCHS is
# read from the YAML file. With K >= EPOCHS the run fits in one job, without --stop-after-epoch and without resumed
# jobs:
#
#   CONFIG=configs/baseline/rfdetr_table7_max_upperbound.yaml; K=<epochs per job>
#   EPOCHS=$(awk '$1 == "epochs:" {print $2}' $CONFIG); NAME=$(basename $CONFIG .yaml)
#   if [ "$K" -ge "$EPOCHS" ]; then
#       sbatch -J $NAME scripts/leonardo/train.sh $CONFIG
#   else
#       J=$(sbatch --parsable -J $NAME scripts/leonardo/train.sh $CONFIG --stop-after-epoch $K)
#       for STOP in $(seq $((2 * K)) $K $((EPOCHS - 1))); do
#           J=$(sbatch --parsable -J $NAME --dependency=afterok:$J scripts/leonardo/train.sh $CONFIG --resume \
#               --stop-after-epoch $STOP)
#       done
#       sbatch -J $NAME --dependency=afterok:$J scripts/leonardo/train.sh $CONFIG --resume
#   fi
#
# and the same with CONFIG=configs/baseline/rfdetr_table7_2xl_upperbound.yaml. A job that still reaches the time limit
# (TIMEOUT) loses the epoch in progress, and the jobs after it never start (afterok). Then:
#   1. scancel the jobs of the chain still in the queue (squeue --me);
#   2. take D, the epochs already done: the last epoch with validation metrics in metrics.csv of output_dir, + 1
#      (Lightning counts epochs from 0);
#   3. launch the chain again as above, but with --resume in every job: the first with --stop-after-epoch $((D + K))
#      (only --resume if D + K >= EPOCHS), the loop over $(seq $((D + 2 * K)) $K $((EPOCHS - 1))), the last with
#      --resume only; with a smaller K, if K epochs did not fit in the job.
#
# Then: memory of the node with `seff <job id>`; W&B synced from the login node, after or during a job:
#
#   wandb sync <output_dir of the YAML>/wandb/offline-run-*
#
# To repeat a run from scratch, delete its output_dir and its W&B run, or pass another id:
# sbatch --export=ALL,WANDB_RUN_ID=<id> ...
set -eo pipefail
source /leonardo_work/IscrC_EoDT/tcentonz/rf-detr-eodt/scripts/leonardo/env.sh

CONFIG=$1
export WANDB_RUN_ID=${WANDB_RUN_ID:-$(basename "$CONFIG" .yaml)}  # the same for every job of the run (§7.3)
export WANDB_RESUME=allow

# srun does not inherit --cpus-per-task from sbatch: pass it, so that each process has its 8 cores (7 workers)
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" python scripts/leonardo/train.py "$@"
