# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""The EoDT changes to RF-DETR's files leave RF-DETR unchanged (step 4 of EODT_SPEC.md §9).

``rfdetr_reference.py`` runs in two subprocesses, one on the fork and one on ``ref-rfdetr`` (same commit,
read-only), and the two dumps must be identical.

Covers:
- ``build_model_from_config`` with an RF-DETR config: same state dict and same output on a fixed image.
- ``build_criterion_from_config``: same loss weights and same losses on fixed targets.
- ``get_param_dict``: same optimizer groups (learning rate, weight decay and parameters of each group).
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

FORK = Path(__file__).resolve().parents[2]
REFERENCE = FORK.parent / "ref-rfdetr"
SCRIPT = Path(__file__).with_name("rfdetr_reference.py")


def _dump_both(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the reference script on the fork and on ref-rfdetr, in parallel, and load the two dumps.

    The two runs are independent and most of their time is spent importing ``rfdetr``.
    """
    runs = []
    for source_root in (FORK, REFERENCE):
        output_file = output_dir / f"{source_root.name}.pt"
        # PYTHONPATH comes before the editable install of the fork; no bytecode, so ref-rfdetr stays untouched
        env = {**os.environ, "PYTHONPATH": str(source_root / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT), str(output_file)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        runs.append((source_root, output_file, process))

    dumps = []
    for source_root, output_file, process in runs:
        _, stderr = process.communicate()
        assert process.returncode == 0, stderr
        dump = torch.load(output_file, weights_only=False)
        assert Path(dump["rfdetr_file"]).resolve().is_relative_to((source_root / "src").resolve())
        dumps.append(dump)
    return dumps[0], dumps[1]


def _assert_identical(fork: Any, reference: Any, path: str) -> None:
    """Compare nested dicts, lists, tensors and plain values exactly."""
    assert type(fork) is type(reference), path
    if isinstance(fork, dict):
        assert fork.keys() == reference.keys(), path
        for key in fork:
            _assert_identical(fork[key], reference[key], f"{path}.{key}")
    elif isinstance(fork, (list, tuple)):
        assert len(fork) == len(reference), path
        for index, (fork_item, reference_item) in enumerate(zip(fork, reference)):
            _assert_identical(fork_item, reference_item, f"{path}[{index}]")
    elif isinstance(fork, torch.Tensor):
        assert torch.equal(fork, reference), path
    else:
        assert fork == reference, path


@pytest.mark.skipif(not REFERENCE.exists(), reason=f"{REFERENCE} not found")
def test_rfdetr_unchanged(tmp_path: Path) -> None:
    """RF-DETR Medium: identical weights, outputs, losses and optimizer groups in the fork and in ref-rfdetr."""
    fork, reference = _dump_both(tmp_path)

    assert fork["state_dict"].keys() == reference["state_dict"].keys()
    for key in fork["state_dict"]:
        assert torch.equal(fork["state_dict"][key], reference["state_dict"][key]), key
    assert set(fork["outputs"]) == {"pred_logits", "pred_boxes", "aux_outputs", "enc_outputs"}
    _assert_identical(fork["outputs"], reference["outputs"], "outputs")
    # 4 decoder layers: aux weights _0 to _2, plus the first stage
    assert {"loss_ce_2", "loss_ce_enc"} <= set(fork["weight_dict"])
    _assert_identical(fork["weight_dict"], reference["weight_dict"], "weight_dict")
    assert {"loss_ce", "loss_ce_2", "loss_ce_enc"} <= set(fork["loss_dict"])
    _assert_identical(fork["loss_dict"], reference["loss_dict"], "loss_dict")
    assert len(fork["param_groups"]) > 1
    _assert_identical(fork["param_groups"], reference["param_groups"], "param_groups")
