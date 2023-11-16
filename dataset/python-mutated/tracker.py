"""Tracks skip tensors on a thread."""
from contextlib import contextmanager
import threading
from typing import Dict, Generator, List, Optional, Tuple
from torch import Tensor
from ..checkpoint import is_checkpointing
from ..dependency import fork, join
from ..microbatch import Batch
from ..stream import AbstractStream
from .layout import SkipLayout
from .namespace import Namespace
from .portal import Portal
__all__: List[str] = []

class SkipTracker:
    """Tracks saved skip tensors.

    It will update the given micro-batch in place. This is because when it
    manipulates the underlying skip tensors, the current micro-batch also has
    to be connected with the skip tensors.

    One thread has one skip tracker. Call :func:`current_skip_tracker` to get
    the skip tracker on the current thread.

    """

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.tensors: Dict[Tuple[Namespace, str], Optional[Tensor]] = {}

    def save(self, batch: Batch, ns: Namespace, name: str, tensor: Optional[Tensor]) -> None:
        if False:
            print('Hello World!')
        self.tensors[ns, name] = tensor

    def load(self, batch: Batch, ns: Namespace, name: str) -> Optional[Tensor]:
        if False:
            while True:
                i = 10
        return self.tensors.pop((ns, name))

    def copy(self, batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream, ns: Namespace, name: str) -> None:
        if False:
            print('Hello World!')
        raise TypeError('copy is not supported for non-portal skip tensors')

class SkipTrackerThroughPotals(SkipTracker):
    """Tracks saved skip tensors through portals. The skip tensors will be
    hidden in portals so that the autograd engine does not need to track them.

    This tracker is only used when the training or evaluating module is wrapped
    with :class:`torchpipe.Pipe`.

    """

    def __init__(self, skip_layout: SkipLayout) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.skip_layout = skip_layout
        self.portals: Dict[Tuple[Namespace, str], Portal] = {}

    def save(self, batch: Batch, ns: Namespace, name: str, tensor: Optional[Tensor]) -> None:
        if False:
            return 10
        'Saves the stashed skip tensor in a portal. The portal is then\n        connected to the given micro-batch with :class:`Join`.\n        '
        if not self.skip_layout.requires_copy(ns, name):
            super().save(batch, ns, name, tensor)
            return
        if (ns, name) not in self.portals:
            if is_checkpointing():
                tensor_life = 3
            else:
                tensor_life = 2
            portal = Portal(tensor, tensor_life)
            self.portals[ns, name] = portal
        else:
            portal = self.portals[ns, name]
            tensor_life = 1
            portal.put_tensor(tensor, tensor_life)
        phony = portal.blue()
        tensor_idx = batch.find_tensor_idx()
        batch[tensor_idx] = join(batch[tensor_idx], phony)

    def load(self, batch: Batch, ns: Namespace, name: str) -> Optional[Tensor]:
        if False:
            return 10
        'Loads a skip tensor from the corresponding portal to pop. The given\n        micro-batch is connected to the portal with :class:`Fork`.\n        '
        if not self.skip_layout.requires_copy(ns, name):
            tensor = super().load(batch, ns, name)
            return tensor
        portal = self.portals[ns, name]
        tensor_idx = batch.find_tensor_idx()
        (batch[tensor_idx], phony) = fork(batch[tensor_idx])
        tensor = portal.orange(phony)
        return tensor

    def copy(self, batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream, ns: Namespace, name: str) -> None:
        if False:
            print('Hello World!')
        'Copies the skip tensor in the corresponding portal. The given\n        micro-batch and the portal will be tied with :class:`Fork` and\n        :class:`Join`.\n        '
        assert self.skip_layout.requires_copy(ns, name)
        tensor_idx = batch.find_tensor_idx()
        (batch[tensor_idx], phony) = fork(batch[tensor_idx])
        portal = self.portals[ns, name]
        phony = portal.copy(prev_stream, next_stream, phony)
        batch[tensor_idx] = join(batch[tensor_idx], phony)

class ThreadLocal(threading.local):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.skip_tracker: Optional[SkipTracker] = None
thread_local = ThreadLocal()

@contextmanager
def use_skip_tracker(skip_tracker: SkipTracker) -> Generator[None, None, None]:
    if False:
        print('Hello World!')
    'Registers the given skip tracker on the current thread within a\n    context::\n\n        with use_skip_tracker(my_skip_tracker):\n            ...\n\n    '
    orig = thread_local.skip_tracker
    thread_local.skip_tracker = skip_tracker
    try:
        yield
    finally:
        thread_local.skip_tracker = orig

def current_skip_tracker() -> SkipTracker:
    if False:
        print('Hello World!')
    'Gets the skip tracker on the current thread.'
    skip_tracker = thread_local.skip_tracker
    if skip_tracker is None:
        skip_tracker = SkipTracker()
        thread_local.skip_tracker = skip_tracker
    return skip_tracker