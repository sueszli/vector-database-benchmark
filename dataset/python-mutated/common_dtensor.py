from contextlib import contextmanager
from dataclasses import dataclass
import itertools
import sys
from functools import wraps
from typing import Any, Callable, Generator, Iterator, Tuple, Dict, List, Sequence, TypeVar, cast
import torch
import torch.distributed as dist
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec
from torch.testing._internal.common_distributed import MultiProcessTestCase, MultiThreadedTestCase, TEST_SKIPS, skip_if_lt_x_gpu
from torch.distributed._tensor import DeviceMesh, Shard, Replicate, distribute_tensor, redistribute
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Placement, DTensorSpec
DEVICE_TYPE = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'cpu'
PG_BACKEND = 'nccl' if DEVICE_TYPE == 'cuda' else 'gloo'
NUM_DEVICES = 4
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    NUM_DEVICES = min(NUM_DEVICES, torch.cuda.device_count())
T = TypeVar('T')

class MLPModule(torch.nn.Module):

    def __init__(self, device):
        if False:
            return 10
        super().__init__()
        torch.manual_seed(5)
        self.net1 = torch.nn.Linear(10, 16, device=device)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(16, 10, device=device)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return self.net2(self.relu(self.net1(x)))

    def reset_parameters(self):
        if False:
            i = 10
            return i + 15
        self.net1.reset_parameters()
        self.net2.reset_parameters()

def skip_unless_torch_gpu(method: T) -> T:
    if False:
        while True:
            i = 10
    "\n    Test decorator which skips the test unless there's a GPU available to torch.\n\n    >>> # xdoctest: +SKIP\n    >>> @skip_unless_torch_gpu\n    >>> def test_some_method(self) -> None:\n    >>>   ...\n    "
    return cast(T, skip_if_lt_x_gpu(NUM_DEVICES)(method))

@dataclass
class RedistributeProfile:
    num_calls: int

@contextmanager
def redistribute_profiler() -> Generator[RedistributeProfile, None, None]:
    if False:
        while True:
            i = 10
    orig_redistribute_local_tensor = redistribute.redistribute_local_tensor
    profile: RedistributeProfile = RedistributeProfile(num_calls=0)

    def patched_redistribute_local_tensor(local_tensor: torch.Tensor, current_spec: DTensorSpec, target_spec: DTensorSpec) -> DTensor:
        if False:
            i = 10
            return i + 15
        result = orig_redistribute_local_tensor(local_tensor, current_spec, target_spec)
        profile.num_calls += 1
        return result
    try:
        redistribute.redistribute_local_tensor = patched_redistribute_local_tensor
        yield profile
    finally:
        redistribute.redistribute_local_tensor = orig_redistribute_local_tensor

class DTensorTestBase(MultiProcessTestCase):

    @property
    def world_size(self) -> int:
        if False:
            i = 10
            return i + 15
        return NUM_DEVICES

    @property
    def backend(self) -> str:
        if False:
            while True:
                i = 10
        return PG_BACKEND

    def build_device_mesh(self) -> DeviceMesh:
        if False:
            return 10
        return DeviceMesh(DEVICE_TYPE, list(range(NUM_DEVICES)))

    def init_pg(self) -> None:
        if False:
            i = 10
            return i + 15
        if 'nccl' in self.backend and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f'multi-gpu-{self.world_size}'].exit_code)
        if self.backend not in ['nccl', 'gloo', 'mpi', 'cpu:gloo,cuda:nccl']:
            raise RuntimeError(f'Backend {self.backend} not supported!')
        dist.init_process_group(backend=self.backend, world_size=self.world_size, rank=self.rank, init_method=f'file://{self.file_name}')
        if 'nccl' in self.backend:
            torch.cuda.set_device(self.rank)

    def destroy_pg(self) -> None:
        if False:
            while True:
                i = 10
        dist.barrier()
        dist.destroy_process_group()

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        super().setUp()
        self._spawn_processes()

    def _test_op(self, mesh: DeviceMesh, op_call, *args, **kwargs) -> None:
        if False:
            return 10
        with redistribute_profiler() as profile:
            out = op_call(*args, **kwargs)
            dtc = DTensorConverter(mesh, args, kwargs)
            for (d_args, d_kwargs) in dtc:
                self.assertEqual(dtc.successful(), True)
                d_out = op_call(*d_args, **d_kwargs)
                self.assertEqual(d_out.redistribute(mesh, [Replicate()] * mesh.ndim).to_local(), out)

    def run_subtests(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return run_subtests(self, *args, **kwargs)
TestFunc = Callable[[object], object]

def with_comms(func: TestFunc) -> TestFunc:
    if False:
        i = 10
        return i + 15
    assert func is not None

    @wraps(func)
    def wrapper(self, *args: Tuple[object], **kwargs: Dict[str, Any]) -> None:
        if False:
            return 10
        if torch.cuda.is_available() and torch.cuda.device_count() >= self.world_size:
            self.device_type = 'cuda'
        else:
            self.device_type = 'cpu'
        self.init_pg()
        func(self, *args, **kwargs)
        self.destroy_pg()
    return wrapper

def run_subtests(cls_inst, subtest_config: Dict[str, List[Any]], test_fn: Callable, *test_args, **test_kwargs: Any):
    if False:
        while True:
            i = 10
    '\n    Runs a test function given by ``test_fn`` as a subtest according to the\n    configurations specified by ``subtest_config``. This amortizes the\n    costly setup overhead (including process spawn and initializing the\n    process group) over the subtests.\n\n    Args:\n        subtest_config (Dict[str, List[Any]]): A mapping from subtest\n            keyword argument name to a list of its possible values.\n        test_fn (Callable): A callable that runs the actual test.\n        test_args: Positional arguments to pass to ``test_fn``.\n        test_kwargs: Keyword arguments to pass to ``test_fn``.\n    '
    subtest_config_items: List[Tuple[str, List[Any]]] = list(subtest_config.items())
    subtest_config_keys: List[str] = [item[0] for item in subtest_config_items]
    subtest_config_values: List[List[Any]] = [item[1] for item in subtest_config_items]
    for values in itertools.product(*subtest_config_values):
        subtest_kwargs = dict(zip(subtest_config_keys, values))
        with cls_inst.subTest(**subtest_kwargs):
            test_fn(*test_args, **test_kwargs, **subtest_kwargs)
        dist.barrier()

class DTensorOpTestBase(MultiThreadedTestCase):

    @property
    def world_size(self) -> int:
        if False:
            i = 10
            return i + 15
        return NUM_DEVICES

    @property
    def device_type(self) -> str:
        if False:
            return 10
        return DEVICE_TYPE

    def build_device_mesh(self):
        if False:
            return 10
        return DeviceMesh(self.device_type, list(range(self.world_size)))

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self._spawn_threads()

class DTensorConverter:

    def __init__(self, mesh: DeviceMesh, args: Tuple[object, ...], kwargs: Dict[str, object]) -> None:
        if False:
            while True:
                i = 10
        self.hit = 0
        self.miss = 0
        self.mesh = mesh
        self.args = args
        self.kwargs = kwargs
        (flatten_args, flatten_args_spec) = tree_flatten(args)
        (flatten_kwargs, flatten_kwargs_spec) = tree_flatten(kwargs)
        self.flatten_args: List[object] = flatten_args
        self.flatten_args_spec: TreeSpec = flatten_args_spec
        self.flatten_kwargs: List[object] = flatten_kwargs
        self.flatten_kwargs_spec: TreeSpec = flatten_kwargs_spec
        choices_for_args = []
        for arg in self.flatten_args:
            if isinstance(arg, torch.Tensor):
                choices_for_args.append(self.gen_sharding_choices_for_arg(arg))
        for arg in self.flatten_kwargs:
            if isinstance(arg, torch.Tensor):
                choices_for_args.append(self.gen_sharding_choices_for_arg(arg))
        self.sharding_combs: Iterator[Sequence[Placement]] = iter(itertools.product(*choices_for_args))

    def successful(self) -> bool:
        if False:
            print('Hello World!')
        return self.hit > 0 and self.miss == 0

    def is_supported_tensor(self, t: torch.Tensor) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return not any([t.is_sparse_csr, t.is_sparse, t.is_mkldnn, t.is_quantized, t.is_nested, torch._is_functional_tensor(t), t.is_neg(), t.is_conj(), t.device.type in ('lazy', 'meta')])

    def gen_sharding_choices_for_arg(self, arg: torch.Tensor) -> Sequence[Placement]:
        if False:
            print('Hello World!')
        mesh_size = self.mesh.size()
        sharding_choices: List[Placement] = [Replicate()]
        if arg.dtype != torch.bool:
            sharding_choices = sharding_choices + [Shard(i) for (i, s) in enumerate(arg.shape) if s > 1 and s % mesh_size == 0]
        return sharding_choices

    def __iter__(self) -> 'DTensorConverter':
        if False:
            return 10
        return self

    def __next__(self) -> Tuple[Tuple[object, ...], Dict[str, object]]:
        if False:
            return 10
        try:
            next_sharding_choices = next(self.sharding_combs)
            idx = 0
            new_args: List[object] = []
            for arg in self.flatten_args:
                if isinstance(arg, torch.Tensor):
                    new_args.append(self.to_dist_tensor(arg, self.mesh, [next_sharding_choices[idx]]))
                    idx += 1
                else:
                    new_args.append(arg)
            new_kwargs: List[object] = []
            for arg in self.flatten_kwargs:
                if isinstance(arg, torch.Tensor):
                    new_kwargs.append(self.to_dist_tensor(arg, self.mesh, [next_sharding_choices[idx]]))
                    idx += 1
                else:
                    new_kwargs.append(arg)
            return (tree_unflatten(new_args, self.flatten_args_spec), tree_unflatten(new_kwargs, self.flatten_kwargs_spec))
        except StopIteration as e:
            raise StopIteration from e

    def to_dist_tensor(self, t: torch.Tensor, mesh: DeviceMesh, placements: List[Placement]) -> torch.Tensor:
        if False:
            return 10
        if type(t) is torch.Tensor or type(t) is torch.nn.Parameter:
            if self.is_supported_tensor(t):
                self.hit += 1
                if t.ndim == 0:
                    r = distribute_tensor(t, mesh, [Replicate()] * mesh.ndim)
                else:
                    r = distribute_tensor(t, mesh, placements)
                if type(t) is torch.nn.Parameter:
                    r = torch.nn.Parameter(r, requires_grad=r.requires_grad)
                return r
            else:
                self.miss += 1
                return t
        elif torch.overrides.is_tensor_like(t):
            self.miss += 1
            return t
        else:
            raise RuntimeError(f'Trying to convert to DTensor, but got {type(t)}')