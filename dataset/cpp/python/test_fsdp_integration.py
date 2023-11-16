# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from copy import deepcopy
from pathlib import Path
from unittest import mock

import pytest
import torch
from lightning.fabric import Fabric
from lightning.fabric.plugins import FSDPPrecision
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0, _TORCH_GREATER_EQUAL_2_1
from lightning.fabric.wrappers import _FabricOptimizer
from torch.distributed.fsdp import FlatParameter, FullyShardedDataParallel, OptimStateKeyType
from torch.distributed.fsdp.wrap import always_wrap_policy, wrap
from torch.nn import Parameter

from tests_fabric.helpers.models import BoringFabric
from tests_fabric.helpers.runif import RunIf
from tests_fabric.test_fabric import BoringModel


class _MyFabric(BoringFabric):
    def get_model(self):
        model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        self.num_wrapped = 4
        return model

    def step(self, model, batch):
        wrapped_layers = [m for m in model.modules() if isinstance(m, FullyShardedDataParallel)]
        assert len(wrapped_layers) == self.num_wrapped
        assert (self.num_wrapped == 4) == isinstance(model._forward_module, FullyShardedDataParallel)

        precision = self._precision
        assert isinstance(precision, FSDPPrecision)
        if precision.precision == "16-mixed":
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.float16
        elif precision.precision == "bf16-mixed":
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.bfloat16
        elif precision.precision == "16-true":
            param_dtype = reduce_dtype = buffer_dtype = torch.float16
        elif precision.precision == "bf16-true":
            param_dtype = reduce_dtype = buffer_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown precision {precision.precision}")

        for layer in wrapped_layers:
            assert layer.mixed_precision.param_dtype == param_dtype
            assert layer.mixed_precision.reduce_dtype == reduce_dtype
            assert layer.mixed_precision.buffer_dtype == buffer_dtype

        output = model(batch)
        return torch.nn.functional.mse_loss(output, torch.ones_like(output))


class _MyFabricManualWrapping(_MyFabric):
    def get_model(self):
        model = super().get_model()
        for i, layer in enumerate(model):
            if i % 2 == 0:
                model[i] = wrap(layer)
        self.num_wrapped = 2
        return model


@RunIf(min_cuda_gpus=2, standalone=True, min_torch="2.0.0")
@pytest.mark.parametrize("precision", ["16-mixed", pytest.param("bf16-mixed", marks=RunIf(bf16_cuda=True))])
@pytest.mark.parametrize("manual_wrapping", [True, False])
def test_fsdp_train_save_load(tmp_path, manual_wrapping, precision):
    """Test FSDP training, saving and loading with different wrapping and precision settings."""
    fabric_cls = _MyFabricManualWrapping if manual_wrapping else _MyFabric
    fabric = fabric_cls(
        accelerator="cuda", strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy), devices=2, precision=precision
    )
    fabric.run()

    checkpoint_path = fabric.broadcast(str(tmp_path / "fsdp-checkpoint"))

    params_before = deepcopy(list(fabric.model.parameters()))
    state = {"model": fabric.model, "optimizer": fabric.optimizer, "steps": 1}
    fabric.save(checkpoint_path, state)
    assert set(os.listdir(checkpoint_path)) == {"meta.pt", ".metadata", "__0_0.distcp", "__1_0.distcp"}

    # re-init all objects and resume
    fabric = fabric_cls(
        accelerator="cuda", strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy), devices=2, precision=precision
    )
    fabric.run()

    # check correctness with loaded state
    state = {"model": fabric.model, "optimizer": fabric.optimizer, "steps": 0}
    metadata = fabric.load(checkpoint_path, state)
    for p0, p1 in zip(params_before, fabric.model.parameters()):
        torch.testing.assert_close(p0, p1, atol=0, rtol=0, equal_nan=True)

    # check user data in state reloaded
    assert state["steps"] == 1
    assert not metadata

    # attempt to load a key not in the metadata checkpoint
    state = {"model": fabric.model, "coconut": 11}
    with pytest.raises(KeyError, match="The requested state contains a key 'coconut' that does not exist"):
        fabric.load(checkpoint_path, state)

    # `strict=False` ignores the missing key
    state = {"model": fabric.model, "coconut": 11}
    fabric.load(checkpoint_path, state, strict=False)
    assert state["coconut"] == 11


@RunIf(min_cuda_gpus=2, standalone=True, min_torch="2.0.0")
def test_fsdp_save_full_state_dict(tmp_path):
    """Test that FSDP saves the full state into a single file with `state_dict_type="full"`."""
    fabric = BoringFabric(
        accelerator="cuda",
        strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy, state_dict_type="full"),
        devices=2,
    )
    fabric.run()

    checkpoint_path = Path(fabric.broadcast(str(tmp_path / "fsdp-checkpoint.pt")))

    state = {"model": fabric.model, "optimizer": fabric.optimizer, "steps": 1}
    fabric.save(checkpoint_path, state)

    checkpoint = torch.load(checkpoint_path)
    assert checkpoint["steps"] == 1
    loaded_state_dict = checkpoint["model"]

    # assert the correct state model was saved
    with FullyShardedDataParallel.summon_full_params(fabric.model):
        state_dict = fabric.model.state_dict()
        assert set(loaded_state_dict.keys()) == set(state_dict.keys())
        for param_name in state_dict:
            assert torch.equal(loaded_state_dict[param_name], state_dict[param_name].cpu())
        params_before = [p.cpu() for p in fabric.model.parameters()]

    # assert the correct optimizer state was saved
    optimizer_state_before = FullyShardedDataParallel.full_optim_state_dict(
        fabric.model, fabric.optimizer, rank0_only=False
    )
    assert set(checkpoint["optimizer"].keys()) == set(optimizer_state_before.keys()) == {"state", "param_groups"}

    # 1. verify the FSDP state can be loaded back into a FSDP model/strategy directly
    fabric = BoringFabric(
        accelerator="cuda",
        strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy),
        devices=2,
    )
    fabric.run()
    metadata = fabric.load(checkpoint_path, {"model": fabric.model, "optimizer": fabric.optimizer})
    assert metadata == {"steps": 1}

    with FullyShardedDataParallel.summon_full_params(fabric.model):
        params_after = list(fabric.model.parameters())
        assert all(torch.equal(p0.cpu(), p1.cpu()) for p0, p1 in zip(params_before, params_after))

    # assert the correct optimizer state was loaded
    optimizer_state_after = FullyShardedDataParallel.full_optim_state_dict(
        fabric.model, fabric.optimizer, rank0_only=False
    )
    assert set(optimizer_state_after.keys()) == set(optimizer_state_before.keys()) == {"state", "param_groups"}
    torch.testing.assert_close(optimizer_state_after["state"], optimizer_state_before["state"], atol=0, rtol=0)
    assert optimizer_state_after["param_groups"] == optimizer_state_before["param_groups"]

    # run a step to verify the optimizer state is correct
    fabric.run()

    # 2. verify the FSDP state can be loaded back into a single-device model/strategy
    fabric = BoringFabric(accelerator="cpu", devices=1)
    fabric.run()
    metadata = fabric.load(checkpoint_path, {"model": fabric.model, "optimizer": fabric.optimizer})
    assert metadata == {"steps": 1}
    params_after = list(fabric.model.parameters())
    assert all(torch.equal(p0, p1) for p0, p1 in zip(params_before, params_after))

    # get optimizer state after loading
    normal_checkpoint_path = Path(fabric.broadcast(str(tmp_path / "normal-checkpoint.pt")))
    fabric.save(normal_checkpoint_path, {"model": fabric.model, "optimizer": fabric.optimizer, "steps": 2})
    optimizer_state_after = torch.load(normal_checkpoint_path)["optimizer"]
    optimizer_state_after = FullyShardedDataParallel.rekey_optim_state_dict(
        optimizer_state_after, optim_state_key_type=OptimStateKeyType.PARAM_NAME, model=fabric.model
    )

    # assert the correct optimizer state was loaded
    assert set(optimizer_state_after.keys()) == set(optimizer_state_before.keys()) == {"state", "param_groups"}
    torch.testing.assert_close(optimizer_state_after["state"], optimizer_state_before["state"], atol=0, rtol=0)

    # run a step to verify the optimizer state is correct
    fabric.run()

    # 3. verify that a single-device model/strategy states can be loaded into a FSDP model/strategy
    fabric = BoringFabric(
        accelerator="cuda",
        strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy),
        devices=2,
    )
    fabric.run()
    metadata = fabric.load(normal_checkpoint_path, {"model": fabric.model, "optimizer": fabric.optimizer})
    assert metadata == {"steps": 2}

    with FullyShardedDataParallel.summon_full_params(fabric.model):
        params_after = list(fabric.model.parameters())
        assert all(torch.equal(p0.cpu(), p1.cpu()) for p0, p1 in zip(params_before, params_after))

    # assert the correct optimizer state was loaded
    optimizer_state_after = FullyShardedDataParallel.full_optim_state_dict(
        fabric.model, fabric.optimizer, rank0_only=False
    )
    assert set(optimizer_state_after.keys()) == set(optimizer_state_before.keys()) == {"state", "param_groups"}
    torch.testing.assert_close(optimizer_state_after["state"], optimizer_state_before["state"], atol=0, rtol=0)
    assert optimizer_state_after["param_groups"] == optimizer_state_before["param_groups"]

    # run a step to verify the optimizer state is correct
    fabric.run()


@RunIf(min_cuda_gpus=2, standalone=True, min_torch="2.0.0")
def test_fsdp_load_full_state_dict_into_sharded_model(tmp_path):
    """Test that the strategy can load a full-state checkpoint into a FSDP sharded model."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    fabric = BoringFabric(accelerator="cuda", devices=1)
    fabric.seed_everything(0)
    fabric.run()

    # Save a full-state-dict checkpoint
    checkpoint_path = Path(fabric.broadcast(str(tmp_path / "full-checkpoint.pt")))
    state = {"model": fabric.model, "optimizer": fabric.optimizer, "steps": 1}
    fabric.save(checkpoint_path, state)

    # Gather all weights and store a copy manually
    with FSDP.summon_full_params(fabric.model, writeback=False, rank0_only=False):
        params_before = torch.cat([p.cpu().view(-1) for p in fabric.model.parameters()])

    # Create a FSDP sharded model
    fabric = BoringFabric(
        accelerator="cuda",
        strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy),
        devices=2,
    )
    fabric.run()

    state = {"model": fabric.model, "optimizer": fabric.optimizer, "steps": 44}
    fabric.load(checkpoint_path, state)
    assert state["steps"] == 1

    # Gather all weights and compare
    with FSDP.summon_full_params(fabric.model, writeback=False, rank0_only=False):
        params_after = torch.cat([p.cpu().view(-1) for p in fabric.model.parameters()])
    assert torch.equal(params_before, params_after)

    # Create a raw state-dict checkpoint to test `Fabric.load_raw` too
    raw_checkpoint_path = checkpoint_path.with_name("model-state-dict")
    if fabric.global_rank == 0:
        checkpoint = torch.load(checkpoint_path)
        torch.save(checkpoint["model"], raw_checkpoint_path)
    fabric.barrier()

    fabric.run()
    fabric.load_raw(raw_checkpoint_path, fabric.model)

    # Gather all weights and compare
    with FSDP.summon_full_params(fabric.model, writeback=False, rank0_only=False):
        params_after = torch.cat([p.cpu().view(-1) for p in fabric.model.parameters()])
    assert torch.equal(params_before, params_after)


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
@pytest.mark.parametrize("move_to_device", [True, False])
@mock.patch("lightning.fabric.wrappers._FabricModule")
def test_setup_module_move_to_device(fabric_module_mock, move_to_device):
    """Test that `move_to_device` does nothing, FSDP decides which device parameters get moved to which device
    (sharding)."""
    strategy = FSDPStrategy(auto_wrap_policy=always_wrap_policy)
    fabric = Fabric(accelerator="cuda", devices=2, strategy=strategy)
    fabric.launch()

    model = torch.nn.Linear(10, 10, bias=False)  # total params: 10 * 10 = 100
    fabric_model = fabric.setup_module(model, move_to_device=move_to_device)
    fabric_module_mock.assert_not_called()

    assert len(list(fabric_model.parameters())) == 1
    # the linear layer got sharded and each part is on the expected device
    assert next(fabric_model.parameters()).device == torch.device("cuda", fabric.local_rank)
    assert next(fabric_model.parameters()).numel() == 50
    if _TORCH_GREATER_EQUAL_2_0:
        # In PyTorch >= 2.0 we set `use_orig_params=True` and don't see flattened parameters
        assert isinstance(next(fabric_model.parameters()), Parameter)
    else:
        assert isinstance(next(fabric_model.parameters()), FlatParameter)

    # The _DeviceDtypeModuleMixin currently can't represent the device in a meaningful way for sharded models
    assert fabric_model.device == torch.device("cpu")
    assert fabric.device == torch.device("cuda", fabric.local_rank)


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="2.0.0")
def test_setup_with_orig_params_and_multiple_param_groups():
    """Test that Fabric sets `use_orig_params` for the user when jointly setting up model and optimizer."""
    strategy = FSDPStrategy(auto_wrap_policy=always_wrap_policy)
    fabric = Fabric(accelerator="cuda", devices=2, strategy=strategy)
    fabric.launch()

    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10, bias=False),
        torch.nn.Linear(5, 2, bias=False),
    )
    optimizer = torch.optim.Adam(
        [
            {"params": model[0].parameters(), "lr": 1e-2},
            {"params": model[1].parameters(), "lr": 1e-6},
        ]
    )

    # set up model and optimizer jointly
    wrapped_model, wrapped_optimizer = fabric.setup(model, optimizer)

    assert fabric.strategy._fsdp_kwargs["use_orig_params"]
    assert isinstance(wrapped_optimizer, _FabricOptimizer)
    assert len(wrapped_optimizer.param_groups) == 2
    for i in range(2):
        layer = wrapped_model._forward_module.module[i]
        assert isinstance(layer, FullyShardedDataParallel)
        assert torch.equal(wrapped_optimizer.param_groups[i]["params"][0], layer.weight)

        # A regular parameter as a view into the flattened parameters
        assert isinstance(layer.weight, torch.nn.Parameter)
        assert not isinstance(layer.weight, FlatParameter)


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, dynamo=True)
@mock.patch.dict(os.environ, {})
@pytest.mark.parametrize(
    "compile_after_setup",
    [
        False,
        # https://github.com/pytorch/pytorch/issues/97811
        pytest.param(True, marks=RunIf(min_python="3.9")),
    ],
)
def test_compile(compile_after_setup):
    """Test that the model can be compiled before and after the model is wrapped in FSDP."""
    model = BoringModel()
    strategy = FSDPStrategy(auto_wrap_policy=always_wrap_policy)
    fabric = Fabric(accelerator="cuda", devices=2, strategy=strategy)
    fabric.launch()

    if not compile_after_setup:
        model = torch.compile(model)

    model = fabric.setup(model)

    if compile_after_setup:
        model = torch.compile(model)

    for _ in range(3):
        model(torch.rand(2, 32, device=fabric.device)).sum().backward()


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("32-true", torch.float32),
        ("16-true", torch.float16),
        pytest.param("bf16-true", torch.bfloat16, marks=RunIf(bf16_cuda=True)),
    ],
)
def test_module_init_context(precision, expected_dtype):
    """Test that the module under the init-context gets moved to the right device and dtype."""
    fabric = Fabric(
        accelerator="cuda",
        devices=2,
        strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy),
        precision=precision,
    )
    fabric.launch()

    def _run_setup_assertions(empty_init, expected_device):
        with fabric.init_module(empty_init=empty_init):
            model = torch.nn.Linear(100, 100, bias=False)

        # The model is on the CPU/meta-device until after `.setup()``
        assert model.weight.device == expected_device
        assert model.weight.dtype == expected_dtype
        model = fabric.setup(model)
        # Parameters get sharded in `.setup()` and moved to the target device
        assert model.weight.device == torch.device("cuda", fabric.local_rank)
        assert model.weight.dtype == expected_dtype

    # Case 1: No empty init
    _run_setup_assertions(empty_init=False, expected_device=torch.device("cpu"))

    if _TORCH_GREATER_EQUAL_2_1:
        # Case 2: Empty-init with PyTorch >= 2.1 supports meta device
        _run_setup_assertions(empty_init=True, expected_device=torch.device("meta"))
    else:
        # Case 2: Empty-init with PyTorch < 2.1 only supports `torch.empty()`-init
        _run_setup_assertions(empty_init=True, expected_device=torch.device("cpu"))


@RunIf(min_cuda_gpus=2, standalone=True, min_torch="2.0.0")
def test_fsdp_save_filter(tmp_path):
    fabric = BoringFabric(accelerator="cuda", strategy=FSDPStrategy(state_dict_type="full"), devices=2)
    fabric.launch()
    model = fabric.get_model()
    model = fabric.setup_module(model)

    tmp_path = Path(fabric.broadcast(str(tmp_path)))
    state = {"model": model}
    filter = {"model": lambda k, v: "bias" in k}

    checkpoint_path = tmp_path / "full.pth"
    fabric.save(checkpoint_path, state, filter=filter)
    checkpoint = torch.load(checkpoint_path)["model"]
    assert set(checkpoint) == {"bias"}
    assert isinstance(checkpoint["bias"], torch.Tensor)

    fabric.strategy._state_dict_type = "sharded"
    checkpoint_path = tmp_path / "sharded"
    with pytest.raises(NotImplementedError, match="doesn't support loading sharded filtered"):
        fabric.save(checkpoint_path, state, filter=filter)


@RunIf(min_torch="1.13", min_cuda_gpus=1)
def test_fsdp_manual_activation_checkpointing():
    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Linear(1, 1))
    strategy = FSDPStrategy(activation_checkpointing_policy={torch.nn.Linear})
    fabric = Fabric(devices=1, accelerator="cuda", strategy=strategy)
    fabric.launch()

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointWrapper,
        apply_activation_checkpointing,
    )

    # manually apply activation checkpointing
    apply_activation_checkpointing(model)

    wrappers = {name for name, mod in model.named_modules() if isinstance(mod, CheckpointWrapper)}
    assert wrappers == {"0", "1"}

    # let fabric set up the model, it shouldn't apply activation checkpointing again
    with pytest.warns(match="is configured, but the model already contains checkpointed"):
        model = fabric.setup(model)

    wrappers = {name for name, mod in model._forward_module.named_modules() if isinstance(mod, CheckpointWrapper)}
    assert wrappers == {"_fsdp_wrapped_module.0", "_fsdp_wrapped_module.1"}


@RunIf(min_cuda_gpus=1)
def test_rewrap_warnings():
    from torch.distributed.fsdp import FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import wrap

    strategy = FSDPStrategy(auto_wrap_policy={torch.nn.Linear})
    fabric = Fabric(devices=1, accelerator="cuda", strategy=strategy)
    fabric.launch()
    with fabric.init_module():
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU(), wrap(torch.nn.Linear(1, 1)))
    with pytest.warns(match="the model is already wrapped"):
        model = fabric.setup(model)
    assert not isinstance(model._forward_module, FullyShardedDataParallel)
    assert isinstance(model._forward_module[2], FullyShardedDataParallel)

    if not _TORCH_GREATER_EQUAL_2_1:
        return

    with fabric.init_module(empty_init=True):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU(), wrap(torch.nn.Linear(1, 1)))
    assert model[0].weight.is_meta
    with pytest.warns(match="there are still parameters on the meta device"):
        fabric_model = fabric.setup(model)
    assert next(fabric_model.parameters()).is_meta
