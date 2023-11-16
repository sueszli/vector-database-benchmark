import itertools
import warnings
from enum import auto, Enum
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp._common_utils import _FSDPState, _get_param_to_fqns
from torch.distributed.fsdp._flat_param import FlatParamHandle

class _ExecOrderWarnStatus(Enum):
    """Used internally for execution order validation."""
    NONE = auto()
    WARNING = auto()
    WARNED = auto()

class _ExecOrderData:
    """
    This contains the data structures to track the execution order. We track
    the pre-forward order on the *first* iteration for forward prefetching
    (which thus assumes static graph) and the post-forward order on *every*
    iteration for backward prefetching (which thus does not assume static
    graph but may be provide an incorrect order).
    """

    def __init__(self, debug_level: dist.DebugLevel, backward_prefetch_limit: int, forward_prefetch_limit: int) -> None:
        if False:
            return 10
        self.handles_pre_forward_order: List[FlatParamHandle] = []
        self.handles_post_forward_order: List[Optional[FlatParamHandle]] = []
        self._iter = 0
        self._backward_prefetch_limit = backward_prefetch_limit
        self._forward_prefetch_limit = forward_prefetch_limit
        self._checking_order: bool = debug_level == dist.DebugLevel.DETAIL
        self.process_group: Optional[dist.ProcessGroup] = None
        self.world_size: Optional[int] = None
        self.all_handles: List[FlatParamHandle] = []
        self.param_to_fqn: Dict[nn.Parameter, List[str]] = {}
        self.current_order_index = 0
        self.warn_status = _ExecOrderWarnStatus.NONE

    def init(self, state: _FSDPState, root_module: nn.Module, process_group: dist.ProcessGroup) -> None:
        if False:
            while True:
                i = 10
        '\n        Initializes the data structures needed for checking the forward order.\n        This should be called after a root FSDP instance has been set during\n        lazy initialization.\n        '
        self.process_group = process_group
        self.rank = process_group.rank()
        self.world_size = process_group.size()
        for handle in traversal_utils._get_fsdp_handles(root_module):
            index = len(self.all_handles)
            self.all_handles.append(handle)
            handle._handle_index = index
        self.param_to_fqn = _get_param_to_fqns(root_module)

    @property
    def is_first_iter(self) -> bool:
        if False:
            print('Hello World!')
        return self._iter == 0

    def get_handle_to_backward_prefetch(self, current_handle: FlatParamHandle) -> Optional[FlatParamHandle]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a :class:`list` of the handles keys of the handles to backward\n        prefetch given the current handles key. If there are no valid handles\n        keys to prefetch, then this returns an empty :class:`list`.\n        '
        current_index = current_handle._post_forward_index
        if current_index is None:
            return None
        target_index = current_index - 1
        target_handle: Optional[FlatParamHandle] = None
        for _ in range(self._backward_prefetch_limit):
            if target_index < 0:
                break
            target_handle = self.handles_post_forward_order[target_index]
            target_index -= 1
        return target_handle

    def get_handle_to_forward_prefetch(self, current_handle: FlatParamHandle) -> Optional[FlatParamHandle]:
        if False:
            print('Hello World!')
        '\n        Returns a :class:`list` of the handles keys of the handles to forward\n        prefetch given the current handles key. If there are no valid handles\n        keys to prefetch, then this returns an empty :class:`list`.\n        '
        current_index = current_handle._pre_forward_order_index
        if current_index is None:
            return None
        target_index = current_index + 1
        target_handle: Optional[FlatParamHandle] = None
        for _ in range(self._forward_prefetch_limit):
            if target_index >= len(self.handles_pre_forward_order):
                break
            target_handle = self.handles_pre_forward_order[target_index]
            target_index += 1
        return target_handle

    def record_post_forward(self, handle: Optional[FlatParamHandle]) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Records ``handles`` in the post-forward order, where ``handles`` should\n        be a group of handles used in the same module's forward. If ``handles``\n        is empty, then it is omitted.\n\n        Unlike :meth:`record_pre_forward`, this records the order *every*\n        iteration with the expectation that the recorded order is reset in\n        :meth:`next_iter`.\n        "
        if not handle:
            return
        if handle._post_forward_index:
            self.handles_post_forward_order.append(handle)
            return
        index = len(self.handles_post_forward_order)
        handle._post_forward_index = index
        self.handles_post_forward_order.append(handle)

    def record_pre_forward(self, handle: Optional[FlatParamHandle], is_training: bool) -> None:
        if False:
            return 10
        "\n        Records ``handles`` in the pre-forward order, where ``handles`` should\n        be a group of handles used in the same module's forward. If ``handles``\n        is empty, then it is omitted.\n\n        On the first iteration, this checks the execution order across ranks.\n        See :meth:`_check_order` for details.\n        "
        if not handle:
            return
        self._check_order(handle, is_training)
        if not self.is_first_iter or handle._pre_forward_order_index is not None:
            return
        index = len(self.handles_pre_forward_order)
        handle._pre_forward_order_index = index
        self.handles_pre_forward_order.append(handle)

    def _check_order(self, handle: FlatParamHandle, is_training: bool) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Checks the forward execution order as long as ``is_training`` is\n        ``True`` since checking in eval mode is not supported. This only checks\n        if the distributed debug level is DETAIL.\n\n        - On the first iteration, this uses all-gathers to check that all ranks\n        are all-gathering the same handles and hence ``FlatParameter`` s,\n        raising an error if not.\n        - On subsequent iterations, this checks that each rank is locally\n        consistent with its own forward order from the first iteration, issuing\n        a warning if not. This issues a warning on the first deviating\n        iteration and stops warning thereafter.\n        '
        if not is_training or not self._checking_order:
            return
        if self.is_first_iter:
            msg_prefix = 'Forward order differs across ranks:'
            optional_local_indices: Tuple[Optional[int], ...] = self._get_handle_indices(handle)
            device = handle.device
            num_valid_indices = sum((index is not None for index in optional_local_indices))
            tensor_kwargs: Dict[str, Union[torch.dtype, torch.device]] = {'dtype': torch.int32, 'device': device}
            world_num_valid_indices = torch.zeros(self.world_size, **tensor_kwargs)
            local_num_valid_indices = torch.tensor([num_valid_indices], **tensor_kwargs)
            dist.all_gather_into_tensor(world_num_valid_indices, local_num_valid_indices, group=self.process_group)
            world_num_valid_indices = world_num_valid_indices.cpu()
            assert self.world_size is not None
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                for ((r1, n1), (r2, n2)) in itertools.combinations(((rank, world_num_valid_indices[rank]) for rank in range(self.world_size)), 2):
                    if n1 != n2:
                        raise RuntimeError(f'{msg_prefix} rank {r1} is all-gathering {n1} parameters while rank {r2} is all-gathering {n2} parameters')
            world_indices = torch.zeros(self.world_size * num_valid_indices, **tensor_kwargs)
            local_indices = torch.tensor(optional_local_indices, **tensor_kwargs)
            dist.all_gather_into_tensor(world_indices, local_indices, group=self.process_group)
            world_indices = world_indices.cpu()
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                for ((r1, i1), (r2, i2)) in itertools.combinations(((rank, world_indices[rank * num_valid_indices:(rank + 1) * num_valid_indices]) for rank in range(self.world_size)), 2):
                    if i1 != i2:
                        r1_param_names = self._get_names_from_handle_indices(i1)
                        r2_param_names = self._get_names_from_handle_indices(i2)
                        raise RuntimeError(f'{msg_prefix} rank {r1} is all-gathering parameters for {r1_param_names} while rank {r2} is all-gathering parameters for {r2_param_names}')
        else:
            if self.warn_status == _ExecOrderWarnStatus.WARNED:
                return
            msg_prefix = None
            if self.current_order_index >= len(self.handles_pre_forward_order):
                msg_prefix = 'Expected to not all-gather any more parameters in the forward but trying to all-gather parameters for '
            else:
                expected_handle = self.handles_pre_forward_order[self.current_order_index]
                if expected_handle != handle:
                    expected_param_names = self._get_names_from_handles(expected_handle)
                    msg_prefix = f'Expected to all-gather for {expected_param_names} but trying to all-gather parameters for '
            if msg_prefix is not None:
                param_names = self._get_names_from_handles(handle)
                msg_suffix = f'{param_names}' if param_names else 'a newly-added parameter since construction time'
                warnings.warn(f'Forward order differs from that of the first iteration on rank {self.rank}. Collectives are unchecked and may give incorrect results or hang.\n{msg_prefix}{msg_suffix}')
                self.warn_status = _ExecOrderWarnStatus.WARNING
            self.current_order_index += 1

    def _get_handle_indices(self, handle: FlatParamHandle) -> Tuple[Optional[int], ...]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the handle indices (i.e. indices into ``self.all_handles``)\n        corresponding to the handles in ``handle``. An entry in the\n        returned tuple is ``None`` if the handle is invalid.\n        '
        indices: List[Optional[int]] = []
        if handle:
            indices.append(handle._handle_index)
        return tuple(indices)

    def _get_names_from_handle_indices(self, handle_indices: Tuple[int, ...]) -> List[List[str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a list of FQNs for each handle in ``handle_indices``. If a\n        handle index is invalid, then its FQNs are omitted from the returned\n        list.\n        '
        fqns: List[List[str]] = []
        for index in handle_indices:
            if index is None or index < 0 or index >= len(self.all_handles):
                continue
            handle = self.all_handles[index]
            flat_param = handle.flat_param
            fqns.append(self.param_to_fqn[flat_param])
        return fqns

    def _get_names_from_handles(self, handle: FlatParamHandle) -> List[List[str]]:
        if False:
            while True:
                i = 10
        '\n        Returns a list of FQNs for each handle in ``handles_key``. If a handle\n        is invalid, then its FQNs are omitted from the returned list.\n        '
        fqns: List[List[str]] = []
        if handle:
            flat_param = handle.flat_param
            if flat_param in self.param_to_fqn:
                fqns.append(self.param_to_fqn[flat_param])
        return fqns

    def next_iter(self):
        if False:
            i = 10
            return i + 15
        '\n        Advances the internal data structures per iteration. This should be\n        called in the post-backward callback since that marks the true end of\n        an iteration.\n        '
        self._iter += 1
        self.handles_post_forward_order.clear()
        if self._checking_order:
            self.current_order_index = 0
            if self.warn_status == _ExecOrderWarnStatus.WARNING:
                self.warn_status = _ExecOrderWarnStatus.WARNED