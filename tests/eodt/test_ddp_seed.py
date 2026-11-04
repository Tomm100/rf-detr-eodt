# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the seed of ``RFDETRModelModule`` with DDP (step 6a of EODT_SPEC.md §9).

Lightning builds the ``DistributedSampler`` after ``on_fit_start`` and seeds it from ``PL_GLOBAL_SEED``. RF-DETR
seeds every rank with ``seed + rank``, which also set a different ``PL_GLOBAL_SEED`` per rank: the ranks shuffled
differently and did not split the dataset. These tests run ``on_fit_start`` as 4 ranks and build the sampler as
Lightning does, without starting processes.
"""

import os
from types import SimpleNamespace

import pytest
import torch
from pytorch_lightning.trainer.connectors.data_connector import _get_distributed_sampler
from pytorch_lightning.trainer.states import RunningStage
from torch.utils.data import DataLoader

from rfdetr.config import TrainConfig
from rfdetr.training.module_model import RFDETRModelModule

NUM_RANKS = 4
NUM_IMAGES = 20
SEED = 0


@pytest.fixture(autouse=True)
def restore_seed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the environment variables that ``seed_everything`` sets, so no other test sees them."""
    monkeypatch.delenv("PL_GLOBAL_SEED", raising=False)
    monkeypatch.delenv("PL_SEED_WORKERS", raising=False)


def _rank(rank: int, epoch: int, old_seed: bool = False) -> tuple[int, list[int], int]:
    """Run ``on_fit_start`` as ``rank`` and build its sampler as Lightning does after that hook.

    Args:
        rank: Global rank of the simulated process.
        epoch: Epoch of the sampler.
        old_seed: Restore the behavior before the fix, a ``PL_GLOBAL_SEED`` of ``seed + rank``.

    Returns:
        The sampler seed, the indices of this rank for ``epoch`` and the torch seed of the process.
    """
    train_config = TrainConfig(dataset_dir="unused", seed=SEED)
    RFDETRModelModule.on_fit_start(SimpleNamespace(train_config=train_config, global_rank=rank))
    if old_seed:
        os.environ["PL_GLOBAL_SEED"] = str(SEED + rank)
    loader = DataLoader(list(range(NUM_IMAGES)), shuffle=True)
    sampler = _get_distributed_sampler(
        loader, shuffle=True, overfit_batches=0, mode=RunningStage.TRAINING, num_replicas=NUM_RANKS, rank=rank
    )
    sampler.set_epoch(epoch)
    return sampler.seed, list(sampler), torch.initial_seed()


@pytest.mark.parametrize("epoch", [0, 1])
def test_ranks_split_the_dataset(epoch: int) -> None:
    """Every rank gets the same sampler seed, and the 4 ranks share out the 20 images without overlap."""
    seeds, indices, torch_seeds = zip(*(_rank(rank, epoch) for rank in range(NUM_RANKS)))

    assert set(seeds) == {SEED}
    assert sorted(index for rank_indices in indices for index in rank_indices) == list(range(NUM_IMAGES))
    assert len(set(torch_seeds)) == NUM_RANKS  # model RNG and augmentation stay different per rank


def test_rank_seeds_overlap_before_the_fix() -> None:
    """With the per-rank sampler seed of RF-DETR, the ranks see some images twice and miss others."""
    indices = [_rank(rank, epoch=0, old_seed=True)[1] for rank in range(NUM_RANKS)]

    assert len({index for rank_indices in indices for index in rank_indices}) < NUM_IMAGES
