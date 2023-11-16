"""
`rich.progress` decorator for iterators.

Usage:
>>> from tqdm.rich import trange, tqdm
>>> for i in trange(10):
...     ...
"""
from warnings import warn
from rich.progress import BarColumn, Progress, ProgressColumn, Text, TimeElapsedColumn, TimeRemainingColumn, filesize
from .std import TqdmExperimentalWarning
from .std import tqdm as std_tqdm
__author__ = {'github.com/': ['casperdcl']}
__all__ = ['tqdm_rich', 'trrange', 'tqdm', 'trange']

class FractionColumn(ProgressColumn):
    """Renders completed/total, e.g. '0.5/2.3 G'."""

    def __init__(self, unit_scale=False, unit_divisor=1000):
        if False:
            i = 10
            return i + 15
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        super().__init__()

    def render(self, task):
        if False:
            while True:
                i = 10
        'Calculate common unit for completed and total.'
        completed = int(task.completed)
        total = int(task.total)
        if self.unit_scale:
            (unit, suffix) = filesize.pick_unit_and_suffix(total, ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'], self.unit_divisor)
        else:
            (unit, suffix) = filesize.pick_unit_and_suffix(total, [''], 1)
        precision = 0 if unit == 1 else 1
        return Text(f'{completed / unit:,.{precision}f}/{total / unit:,.{precision}f} {suffix}', style='progress.download')

class RateColumn(ProgressColumn):
    """Renders human readable transfer speed."""

    def __init__(self, unit='', unit_scale=False, unit_divisor=1000):
        if False:
            for i in range(10):
                print('nop')
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        super().__init__()

    def render(self, task):
        if False:
            while True:
                i = 10
        'Show data transfer speed.'
        speed = task.speed
        if speed is None:
            return Text(f'? {self.unit}/s', style='progress.data.speed')
        if self.unit_scale:
            (unit, suffix) = filesize.pick_unit_and_suffix(speed, ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'], self.unit_divisor)
        else:
            (unit, suffix) = filesize.pick_unit_and_suffix(speed, [''], 1)
        precision = 0 if unit == 1 else 1
        return Text(f'{speed / unit:,.{precision}f} {suffix}{self.unit}/s', style='progress.data.speed')

class tqdm_rich(std_tqdm):
    """Experimental rich.progress GUI version of tqdm!"""

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        '\n        This class accepts the following parameters *in addition* to\n        the parameters accepted by `tqdm`.\n\n        Parameters\n        ----------\n        progress  : tuple, optional\n            arguments for `rich.progress.Progress()`.\n        options  : dict, optional\n            keyword arguments for `rich.progress.Progress()`.\n        '
        kwargs = kwargs.copy()
        kwargs['gui'] = True
        kwargs['disable'] = bool(kwargs.get('disable', False))
        progress = kwargs.pop('progress', None)
        options = kwargs.pop('options', {}).copy()
        super(tqdm_rich, self).__init__(*args, **kwargs)
        if self.disable:
            return
        warn('rich is experimental/alpha', TqdmExperimentalWarning, stacklevel=2)
        d = self.format_dict
        if progress is None:
            progress = ('[progress.description]{task.description}[progress.percentage]{task.percentage:>4.0f}%', BarColumn(bar_width=None), FractionColumn(unit_scale=d['unit_scale'], unit_divisor=d['unit_divisor']), '[', TimeElapsedColumn(), '<', TimeRemainingColumn(), ',', RateColumn(unit=d['unit'], unit_scale=d['unit_scale'], unit_divisor=d['unit_divisor']), ']')
        options.setdefault('transient', not self.leave)
        self._prog = Progress(*progress, **options)
        self._prog.__enter__()
        self._task_id = self._prog.add_task(self.desc or '', **d)

    def close(self):
        if False:
            while True:
                i = 10
        if self.disable:
            return
        super(tqdm_rich, self).close()
        self._prog.__exit__(None, None, None)

    def clear(self, *_, **__):
        if False:
            for i in range(10):
                print('nop')
        pass

    def display(self, *_, **__):
        if False:
            while True:
                i = 10
        if not hasattr(self, '_prog'):
            return
        self._prog.update(self._task_id, completed=self.n, description=self.desc)

    def reset(self, total=None):
        if False:
            print('Hello World!')
        '\n        Resets to 0 iterations for repeated use.\n\n        Parameters\n        ----------\n        total  : int or float, optional. Total to use for the new bar.\n        '
        if hasattr(self, '_prog'):
            self._prog.reset(total=total)
        super(tqdm_rich, self).reset(total=total)

def trrange(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Shortcut for `tqdm.rich.tqdm(range(*args), **kwargs)`.'
    return tqdm_rich(range(*args), **kwargs)
tqdm = tqdm_rich
trange = trrange