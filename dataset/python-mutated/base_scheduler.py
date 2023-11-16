from torch.ao.pruning import BaseSparsifier
from functools import wraps
import warnings
import weakref
__all__ = ['BaseScheduler']

class BaseScheduler:

    def __init__(self, sparsifier, last_epoch=-1, verbose=False):
        if False:
            while True:
                i = 10
        if not isinstance(sparsifier, BaseSparsifier):
            raise TypeError(f'{type(sparsifier).__name__} is not an instance of torch.ao.pruning.BaseSparsifier')
        self.sparsifier = sparsifier
        self.base_sl = [group['sparsity_level'] for group in sparsifier.groups]
        self.last_epoch = last_epoch

        def with_counter(method):
            if False:
                return 10
            if getattr(method, '_with_counter', False):
                return method
            instance_ref = weakref.ref(method.__self__)
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                if False:
                    print('Hello World!')
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)
            wrapper._with_counter = True
            return wrapper
        self.sparsifier.step = with_counter(self.sparsifier.step)
        self.sparsifier._step_count = 0
        self._step_count: int = 0
        self.verbose = verbose
        self._get_sl_called_within_step: bool = False
        self.step()

    def state_dict(self):
        if False:
            while True:
                i = 10
        'Returns the state of the scheduler as a :class:`dict`.\n\n        It contains an entry for every variable in self.__dict__ which\n        is not the sparsifier.\n        '
        return {key: value for (key, value) in self.__dict__.items() if key != 'sparsifier'}

    def load_state_dict(self, state_dict):
        if False:
            for i in range(10):
                print('nop')
        'Loads the schedulers state.\n\n        Args:\n            state_dict (dict): scheduler state. Should be an object returned\n                from a call to :meth:`state_dict`.\n        '
        self.__dict__.update(state_dict)

    def get_last_sl(self):
        if False:
            i = 10
            return i + 15
        ' Return last computed sparsity level by current scheduler.\n        '
        return self._last_sl

    def get_sl(self):
        if False:
            return 10
        if not self._get_sl_called_within_step:
            warnings.warn('To get the last sparsity level computed by the scheduler, please use `get_last_sl()`.')
        raise NotImplementedError

    def print_sl(self, is_verbose, group, sl, epoch=None):
        if False:
            for i in range(10):
                print('nop')
        'Display the current sparsity level.\n        '
        if is_verbose:
            if epoch is None:
                print(f'Adjusting sparsity level of group {group} to {sl:.4e}.')
            else:
                print(f'Epoch {epoch:5d}: adjusting sparsity level of group {group} to {sl:.4e}.')

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        format_string = self.__class__.__name__ + ' ('
        format_string += '\n'
        format_string += f'Sparsifier {self.sparsifier}\n'
        format_string += f'    base_sl: {self.base_sl}\n'
        format_string += ')'
        return format_string

    def step(self, epoch=None):
        if False:
            print('Hello World!')
        if self._step_count == 1:
            if not hasattr(self.sparsifier.step, '_with_counter'):
                warnings.warn('Seems like `sparsifier.step()` has been overridden after sparsity scheduler initialization. Please, make sure to call `sparsifier.step()` before `scheduler.step()`.', UserWarning)
            elif self.sparsifier._step_count < 1:
                warnings.warn('Detected call of `scheduler.step()` before `sparsifier.step()`. You have to make sure you run the sparsifier.step() BEFORE any calls to the scheduler.step().', UserWarning)
        self._step_count += 1

        class _enable_get_sl_call:

            def __init__(self, o):
                if False:
                    i = 10
                    return i + 15
                self.o = o

            def __enter__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.o._get_sl_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                if False:
                    i = 10
                    return i + 15
                self.o._get_sl_called_within_step = False
        with _enable_get_sl_call(self):
            self.last_epoch += 1
            values = self.get_sl()
        for (i, data) in enumerate(zip(self.sparsifier.groups, values)):
            (param_group, sl) = data
            param_group['sparsity_level'] = sl
            self.print_sl(self.verbose, i, sl, epoch)
        self._last_sl = [group['sparsity_level'] for group in self.sparsifier.groups]
        self.sparsifier.enable_mask_update = True

    def _make_sure_a_list(self, var):
        if False:
            return 10
        'Utility that extends it to the same length as the .groups, ensuring it is a list'
        n = len(self.sparsifier.groups)
        if not isinstance(var, (list, tuple)):
            return [var] * n
        else:
            if len(var) != n:
                raise ValueError(f'Expected variable of length {n}, but got {len(var)}')
            return list(var)