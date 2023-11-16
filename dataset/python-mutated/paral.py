from functools import partial
from threading import Thread
from typing import Callable, Text, Union
from joblib import Parallel, delayed
from joblib._parallel_backends import MultiprocessingBackend
import pandas as pd
from queue import Queue
import concurrent
from qlib.config import C, QlibConfig

class ParallelExt(Parallel):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        maxtasksperchild = kwargs.pop('maxtasksperchild', None)
        super(ParallelExt, self).__init__(*args, **kwargs)
        if isinstance(self._backend, MultiprocessingBackend):
            self._backend_args['maxtasksperchild'] = maxtasksperchild

def datetime_groupby_apply(df, apply_func: Union[Callable, Text], axis=0, level='datetime', resample_rule='M', n_jobs=-1):
    if False:
        while True:
            i = 10
    'datetime_groupby_apply\n    This function will apply the `apply_func` on the datetime level index.\n\n    Parameters\n    ----------\n    df :\n        DataFrame for processing\n    apply_func : Union[Callable, Text]\n        apply_func for processing the data\n        if a string is given, then it is treated as naive pandas function\n    axis :\n        which axis is the datetime level located\n    level :\n        which level is the datetime level\n    resample_rule :\n        How to resample the data to calculating parallel\n    n_jobs :\n        n_jobs for joblib\n    Returns:\n        pd.DataFrame\n    '

    def _naive_group_apply(df):
        if False:
            return 10
        if isinstance(apply_func, str):
            return getattr(df.groupby(axis=axis, level=level), apply_func)()
        return df.groupby(axis=axis, level=level).apply(apply_func)
    if n_jobs != 1:
        dfs = ParallelExt(n_jobs=n_jobs)((delayed(_naive_group_apply)(sub_df) for (idx, sub_df) in df.resample(resample_rule, axis=axis, level=level)))
        return pd.concat(dfs, axis=axis).sort_index()
    else:
        return _naive_group_apply(df)

class AsyncCaller:
    """
    This AsyncCaller tries to make it easier to async call

    Currently, it is used in MLflowRecorder to make functions like `log_params` async

    NOTE:
    - This caller didn't consider the return value
    """
    STOP_MARK = '__STOP'

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._q = Queue()
        self._stop = False
        self._t = Thread(target=self.run)
        self._t.start()

    def close(self):
        if False:
            while True:
                i = 10
        self._q.put(self.STOP_MARK)

    def run(self):
        if False:
            return 10
        while True:
            data = self._q.get()
            if data == self.STOP_MARK:
                break
            data()

    def __call__(self, func, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._q.put(partial(func, *args, **kwargs))

    def wait(self, close=True):
        if False:
            while True:
                i = 10
        if close:
            self.close()
        self._t.join()

    @staticmethod
    def async_dec(ac_attr):
        if False:
            return 10

        def decorator_func(func):
            if False:
                print('Hello World!')

            def wrapper(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                if isinstance(getattr(self, ac_attr, None), Callable):
                    return getattr(self, ac_attr)(func, self, *args, **kwargs)
                else:
                    return func(self, *args, **kwargs)
            return wrapper
        return decorator_func

class DelayedTask:

    def get_delayed_tuple(self):
        if False:
            while True:
                i = 10
        'get_delayed_tuple.\n        Return the delayed_tuple created by joblib.delayed\n        '
        raise NotImplementedError('NotImplemented')

    def set_res(self, res):
        if False:
            while True:
                i = 10
        'set_res.\n\n        Parameters\n        ----------\n        res :\n            the executed result of the delayed tuple\n        '
        self.res = res

    def get_replacement(self):
        if False:
            for i in range(10):
                print('nop')
        'return the object to replace the delayed task'
        raise NotImplementedError('NotImplemented')

class DelayedTuple(DelayedTask):

    def __init__(self, delayed_tpl):
        if False:
            return 10
        self.delayed_tpl = delayed_tpl
        self.res = None

    def get_delayed_tuple(self):
        if False:
            print('Hello World!')
        return self.delayed_tpl

    def get_replacement(self):
        if False:
            while True:
                i = 10
        return self.res

class DelayedDict(DelayedTask):
    """DelayedDict.
    It is designed for following feature:
    Converting following existing code to parallel
    - constructing a dict
    - key can be gotten instantly
    - computation of values tasks a lot of time.
        - AND ALL the values are calculated in a SINGLE function
    """

    def __init__(self, key_l, delayed_tpl):
        if False:
            while True:
                i = 10
        self.key_l = key_l
        self.delayed_tpl = delayed_tpl

    def get_delayed_tuple(self):
        if False:
            while True:
                i = 10
        return self.delayed_tpl

    def get_replacement(self):
        if False:
            for i in range(10):
                print('nop')
        return dict(zip(self.key_l, self.res))

def is_delayed_tuple(obj) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'is_delayed_tuple.\n\n    Parameters\n    ----------\n    obj : object\n\n    Returns\n    -------\n    bool\n        is `obj` joblib.delayed tuple\n    '
    return isinstance(obj, tuple) and len(obj) == 3 and callable(obj[0])

def _replace_and_get_dt(complex_iter):
    if False:
        for i in range(10):
            print('nop')
    '_replace_and_get_dt.\n\n    FIXME: this function may cause infinite loop when the complex data-structure contains loop-reference\n\n    Parameters\n    ----------\n    complex_iter :\n        complex_iter\n    '
    if isinstance(complex_iter, DelayedTask):
        dt = complex_iter
        return (dt, [dt])
    elif is_delayed_tuple(complex_iter):
        dt = DelayedTuple(complex_iter)
        return (dt, [dt])
    elif isinstance(complex_iter, (list, tuple)):
        new_ci = []
        dt_all = []
        for item in complex_iter:
            (new_item, dt_list) = _replace_and_get_dt(item)
            new_ci.append(new_item)
            dt_all += dt_list
        return (new_ci, dt_all)
    elif isinstance(complex_iter, dict):
        new_ci = {}
        dt_all = []
        for (key, item) in complex_iter.items():
            (new_item, dt_list) = _replace_and_get_dt(item)
            new_ci[key] = new_item
            dt_all += dt_list
        return (new_ci, dt_all)
    else:
        return (complex_iter, [])

def _recover_dt(complex_iter):
    if False:
        for i in range(10):
            print('nop')
    '_recover_dt.\n\n    replace all the DelayedTask in the `complex_iter` with its `.res` value\n\n    FIXME: this function may cause infinite loop when the complex data-structure contains loop-reference\n\n    Parameters\n    ----------\n    complex_iter :\n        complex_iter\n    '
    if isinstance(complex_iter, DelayedTask):
        return complex_iter.get_replacement()
    elif isinstance(complex_iter, (list, tuple)):
        return [_recover_dt(item) for item in complex_iter]
    elif isinstance(complex_iter, dict):
        return {key: _recover_dt(item) for (key, item) in complex_iter.items()}
    else:
        return complex_iter

def complex_parallel(paral: Parallel, complex_iter):
    if False:
        i = 10
        return i + 15
    'complex_parallel.\n    Find all the delayed function created by delayed in complex_iter, run them parallelly and then replace it with the result\n\n    >>> from qlib.utils.paral import complex_parallel\n    >>> from joblib import Parallel, delayed\n    >>> complex_iter = {"a": delayed(sum)([1,2,3]), "b": [1, 2, delayed(sum)([10, 1])]}\n    >>> complex_parallel(Parallel(), complex_iter)\n    {\'a\': 6, \'b\': [1, 2, 11]}\n\n    Parameters\n    ----------\n    paral : Parallel\n        paral\n    complex_iter :\n        NOTE: only list, tuple and dict will be explored!!!!\n\n    Returns\n    -------\n    complex_iter whose delayed joblib tasks are replaced with its execution results.\n    '
    (complex_iter, dt_all) = _replace_and_get_dt(complex_iter)
    for (res, dt) in zip(paral((dt.get_delayed_tuple() for dt in dt_all)), dt_all):
        dt.set_res(res)
    complex_iter = _recover_dt(complex_iter)
    return complex_iter

class call_in_subproc:
    """
    When we repeatedly run functions, it is hard to avoid memory leakage.
    So we run it in the subprocess to ensure it is OK.

    NOTE: Because local object can't be pickled. So we can't implement it via closure.
          We have to implement it via callable Class
    """

    def __init__(self, func: Callable, qlib_config: QlibConfig=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        func : Callable\n            the function to be wrapped\n\n        qlib_config : QlibConfig\n            Qlib config for initialization in subprocess\n\n        Returns\n        -------\n        Callable\n        '
        self.func = func
        self.qlib_config = qlib_config

    def _func_mod(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Modify the initial function by adding Qlib initialization'
        if self.qlib_config is not None:
            C.register_from_C(self.qlib_config)
        return self.func(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            return executor.submit(self._func_mod, *args, **kwargs).result()