# Environment of the EoDT jobs on Leonardo (EODT_SPEC.md §11), sourced by the other scripts of this folder.
# The compute nodes have no internet: the DINOv2 weights are read from HF_HOME and RF-DETR's published checkpoints
# from RF_HOME, both filled on the login node by prepare_offline.sh, and W&B writes offline into the output_dir of the
# run, to be synced from the login node.

WORK_DIR=/leonardo_work/IscrC_EoDT/tcentonz
CODE_DIR=$WORK_DIR/rf-detr-eodt

module load python cuda
source "$WORK_DIR/eodt-venv/bin/activate"
cd "$CODE_DIR"

export HF_HOME=$WORK_DIR/hf_cache  # on $WORK, not on $HOME, which is small
export HF_HUB_OFFLINE=1
export RF_HOME=$WORK_DIR/rfdetr_cache  # RF-DETR's published checkpoints, for the latency benchmark (§6)
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1  # logs reach the SLURM output while the job runs
