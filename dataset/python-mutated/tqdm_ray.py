import builtins
import copy
import json
import logging
import os
import sys
import threading
import uuid
from typing import Any, Dict, Iterable, Optional
import colorama
import ray
from ray._private.ray_constants import env_bool
from ray.util.debug import log_once
try:
    import tqdm.auto as real_tqdm
except ImportError:
    real_tqdm = None
logger = logging.getLogger(__name__)
ProgressBarState = Dict[str, Any]
RAY_TQDM_MAGIC = '__ray_tqdm_magic_token__'
_manager: Optional['_BarManager'] = None
_print = builtins.print

def safe_print(*args, **kwargs):
    if False:
        while True:
            i = 10
    'Use this as an alternative to `print` that will not corrupt tqdm output.\n\n    By default, the builtin print will be patched to this function when tqdm_ray is\n    used. To disable this, set RAY_TQDM_PATCH_PRINT=0.\n    '
    if kwargs.get('file') not in [sys.stdout, sys.stderr, None]:
        return _print(*args, **kwargs)
    try:
        instance().hide_bars()
        _print(*args, **kwargs)
    finally:
        instance().unhide_bars()

class tqdm:
    """Experimental: Ray distributed tqdm implementation.

    This class lets you use tqdm from any Ray remote task or actor, and have the
    progress centrally reported from the driver. This avoids issues with overlapping
    / conflicting progress bars, as the driver centrally manages tqdm positions.

    Supports a limited subset of tqdm args.
    """

    def __init__(self, iterable: Optional[Iterable]=None, desc: Optional[str]=None, total: Optional[int]=None, position: Optional[int]=None):
        if False:
            while True:
                i = 10
        import ray._private.services as services
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except (TypeError, AttributeError):
                total = None
        self._iterable = iterable
        self._desc = desc or ''
        self._total = total
        self._ip = services.get_node_ip_address()
        self._pid = os.getpid()
        self._pos = position or 0
        self._uuid = uuid.uuid4().hex
        self._x = 0
        self._closed = False

    def set_description(self, desc):
        if False:
            i = 10
            return i + 15
        'Implements tqdm.tqdm.set_description.'
        self._desc = desc
        self._dump_state()

    def update(self, n=1):
        if False:
            return 10
        'Implements tqdm.tqdm.update.'
        self._x += n
        self._dump_state()

    def close(self):
        if False:
            print('Hello World!')
        'Implements tqdm.tqdm.close.'
        self._closed = True
        if ray is not None:
            self._dump_state()

    def refresh(self):
        if False:
            while True:
                i = 10
        'Implements tqdm.tqdm.refresh.'
        self._dump_state()

    @property
    def total(self) -> Optional[int]:
        if False:
            return 10
        return self._total

    @total.setter
    def total(self, total: int):
        if False:
            for i in range(10):
                print('nop')
        self._total = total

    def _dump_state(self) -> None:
        if False:
            while True:
                i = 10
        if ray._private.worker.global_worker.mode == ray.WORKER_MODE:
            print(json.dumps(self._get_state()) + '\n', end='')
        else:
            instance().process_state_update(copy.deepcopy(self._get_state()))

    def _get_state(self) -> ProgressBarState:
        if False:
            return 10
        return {'__magic_token__': RAY_TQDM_MAGIC, 'x': self._x, 'pos': self._pos, 'desc': self._desc, 'total': self._total, 'ip': self._ip, 'pid': self._pid, 'uuid': self._uuid, 'closed': self._closed}

    def __iter__(self):
        if False:
            print('Hello World!')
        if self._iterable is None:
            raise ValueError('No iterable provided')
        for x in iter(self._iterable):
            self.update(1)
            yield x

class _Bar:
    """Manages a single virtual progress bar on the driver.

    The actual position of individual bars is calculated as (pos_offset + position),
    where `pos_offset` is the position offset determined by the BarManager.
    """

    def __init__(self, state: ProgressBarState, pos_offset: int):
        if False:
            i = 10
            return i + 15
        'Initialize a bar.\n\n        Args:\n            state: The initial progress bar state.\n            pos_offset: The position offset determined by the BarManager.\n        '
        self.state = state
        self.pos_offset = pos_offset
        self.bar = real_tqdm.tqdm(desc=state['desc'] + ' ' + str(state['pos']), total=state['total'], position=pos_offset + state['pos'], leave=False)
        if state['x']:
            self.bar.update(state['x'])

    def update(self, state: ProgressBarState) -> None:
        if False:
            i = 10
            return i + 15
        'Apply the updated worker progress bar state.'
        if state['desc'] != self.state['desc']:
            self.bar.set_description(state['desc'])
        if state['total'] != self.state['total']:
            self.bar.total = state['total']
            self.bar.refresh()
        delta = state['x'] - self.state['x']
        if delta:
            self.bar.update(delta)
        self.state = state

    def close(self):
        if False:
            print('Hello World!')
        'The progress bar has been closed.'
        self.bar.close()

    def update_offset(self, pos_offset: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Update the position offset assigned by the BarManager.'
        if pos_offset != self.pos_offset:
            self.pos_offset = pos_offset
            self.bar.clear()
            self.bar.pos = -(pos_offset + self.state['pos'])
            self.bar.refresh()

class _BarGroup:
    """Manages a group of virtual progress bar produced by a single worker.

    All the progress bars in the group have the same `pos_offset` determined by the
    BarManager for the process.
    """

    def __init__(self, ip, pid, pos_offset):
        if False:
            return 10
        self.ip = ip
        self.pid = pid
        self.pos_offset = pos_offset
        self.bars_by_uuid: Dict[str, _Bar] = {}

    def has_bar(self, bar_uuid) -> bool:
        if False:
            while True:
                i = 10
        'Return whether this bar exists.'
        return bar_uuid in self.bars_by_uuid

    def allocate_bar(self, state: ProgressBarState) -> None:
        if False:
            while True:
                i = 10
        'Add a new bar to this group.'
        self.bars_by_uuid[state['uuid']] = _Bar(state, self.pos_offset)

    def update_bar(self, state: ProgressBarState) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Update the state of a managed bar in this group.'
        bar = self.bars_by_uuid[state['uuid']]
        bar.update(state)

    def close_bar(self, state: ProgressBarState) -> None:
        if False:
            while True:
                i = 10
        'Remove a bar from this group.'
        bar = self.bars_by_uuid[state['uuid']]
        bar.close()
        del self.bars_by_uuid[state['uuid']]

    def slots_required(self):
        if False:
            while True:
                i = 10
        'Return the number of pos slots we need to accomodate bars in this group.'
        if not self.bars_by_uuid:
            return 0
        return 1 + max((bar.state['pos'] for bar in self.bars_by_uuid.values()))

    def update_offset(self, offset: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Update the position offset assigned by the BarManager.'
        if offset != self.pos_offset:
            self.pos_offset = offset
            for bar in self.bars_by_uuid.values():
                bar.update_offset(offset)

    def hide_bars(self) -> None:
        if False:
            while True:
                i = 10
        'Temporarily hide visible bars to avoid conflict with other log messages.'
        for bar in self.bars_by_uuid.values():
            bar.bar.clear()

    def unhide_bars(self) -> None:
        if False:
            return 10
        'Opposite of hide_bars().'
        for bar in self.bars_by_uuid.values():
            bar.bar.refresh()

class _BarManager:
    """Central tqdm manager run on the driver.

    This class holds a collection of BarGroups and updates their `pos_offset` as
    needed to ensure individual progress bars do not collide in position, kind of
    like a virtual memory manager.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        import ray._private.services as services
        self.ip = services.get_node_ip_address()
        self.pid = os.getpid()
        self.bar_groups = {}
        self.in_hidden_state = False
        self.num_hides = 0
        self.lock = threading.RLock()
        self.should_colorize = not ray.widgets.util.in_notebook()

    def process_state_update(self, state: ProgressBarState) -> None:
        if False:
            return 10
        "Apply the remote progress bar state update.\n\n        This creates a new bar locally if it doesn't already exist. When a bar is\n        created or destroyed, we also recalculate and update the `pos_offset` of each\n        BarGroup on the screen.\n        "
        with self.lock:
            self._process_state_update_locked(state)

    def _process_state_update_locked(self, state: ProgressBarState) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not real_tqdm:
            if log_once('no_tqdm'):
                logger.warning('tqdm is not installed. Progress bars will be disabled.')
            return
        if state['ip'] == self.ip:
            if state['pid'] == self.pid:
                prefix = ''
            else:
                prefix = '(pid={}) '.format(state.get('pid'))
                if self.should_colorize:
                    prefix = '{}{}{}{}'.format(colorama.Style.DIM, colorama.Fore.CYAN, prefix, colorama.Style.RESET_ALL)
        else:
            prefix = '(pid={}, ip={}) '.format(state.get('pid'), state.get('ip'))
            if self.should_colorize:
                prefix = '{}{}{}{}'.format(colorama.Style.DIM, colorama.Fore.CYAN, prefix, colorama.Style.RESET_ALL)
        state['desc'] = prefix + state['desc']
        process = self._get_or_allocate_bar_group(state)
        if process.has_bar(state['uuid']):
            if state['closed']:
                process.close_bar(state)
                self._update_offsets()
            else:
                process.update_bar(state)
        else:
            process.allocate_bar(state)
            self._update_offsets()

    def hide_bars(self) -> None:
        if False:
            print('Hello World!')
        'Temporarily hide visible bars to avoid conflict with other log messages.'
        with self.lock:
            if not self.in_hidden_state:
                self.in_hidden_state = True
                self.num_hides += 1
                for group in self.bar_groups.values():
                    group.hide_bars()

    def unhide_bars(self) -> None:
        if False:
            i = 10
            return i + 15
        'Opposite of hide_bars().'
        with self.lock:
            if self.in_hidden_state:
                self.in_hidden_state = False
                for group in self.bar_groups.values():
                    group.unhide_bars()

    def _get_or_allocate_bar_group(self, state: ProgressBarState):
        if False:
            i = 10
            return i + 15
        ptuple = (state['ip'], state['pid'])
        if ptuple not in self.bar_groups:
            offset = sum((p.slots_required() for p in self.bar_groups.values()))
            self.bar_groups[ptuple] = _BarGroup(state['ip'], state['pid'], offset)
        return self.bar_groups[ptuple]

    def _update_offsets(self):
        if False:
            while True:
                i = 10
        offset = 0
        for proc in self.bar_groups.values():
            proc.update_offset(offset)
            offset += proc.slots_required()

def instance() -> _BarManager:
    if False:
        while True:
            i = 10
    'Get or create a BarManager for this process.'
    global _manager
    if _manager is None:
        _manager = _BarManager()
        if env_bool('RAY_TQDM_PATCH_PRINT', True):
            import builtins
            builtins.print = safe_print
    return _manager
if __name__ == '__main__':
    import time

    @ray.remote
    def processing(delay):
        if False:
            return 10

        def sleep(x):
            if False:
                i = 10
                return i + 15
            print('Intermediate result', x)
            time.sleep(delay)
            return x
        ray.data.range(1000, parallelism=100).map(sleep, compute=ray.data.ActorPoolStrategy(size=1)).count()
    ray.get([processing.remote(0.03), processing.remote(0.01), processing.remote(0.05)])