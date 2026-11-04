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
# Train EoDT or the baseline from its YAML file on one node with 4 A100 (EODT_SPEC.md §9, step 6): one process per
# GPU, launched by srun, which Lightning recognises as a SLURM DDP run. From the code folder on the login node:
#
#   sbatch -J eodt_vits_576_l2-3_learned scripts/leonardo/train.sh configs/eodt/eodt_vits_576_l2-3_learned.yaml
#
# The arguments are those of scripts/leonardo/train.py. A run longer than 24 hours continues in a new job, from
# last.ckpt and in the same W&B run (WANDB_RUN_ID is the name of the YAML file):
#
#   sbatch -J eodt_vits_576_l2-3_learned --dependency=afterany:<job id> scripts/leonardo/train.sh \
#       configs/eodt/eodt_vits_576_l2-3_learned.yaml --resume
#
# Test job, twice in the debug QOS (30 minutes each): the second job resumes the first.
#
#   sbatch -J eodt_testjob --qos=boost_qos_dbg --time=00:30:00 scripts/leonardo/train.sh \
#       configs/eodt/eodt_vits_576_l2-3_learned_testjob.yaml --run-type test \
#       --limit-train-batches 300 --limit-val-batches 40 --stop-after-epoch 1
#   sbatch -J eodt_testjob --qos=boost_qos_dbg --time=00:30:00 --dependency=afterok:<job id> scripts/leonardo/train.sh \
#       configs/eodt/eodt_vits_576_l2-3_learned_testjob.yaml --run-type test \
#       --limit-train-batches 300 --limit-val-batches 40 --resume
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
