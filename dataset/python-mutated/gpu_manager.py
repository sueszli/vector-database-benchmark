"""Module holds Ray actor-class that stores ``cudf.DataFrame``s."""
import cudf
import pandas
import ray
from modin.core.execution.ray.common import RayWrapper

@ray.remote(num_gpus=1)
class GPUManager(object):
    """
    Ray actor-class to store ``cudf.DataFrame``-s and execute functions on it.

    Parameters
    ----------
    gpu_id : int
        The identifier of GPU.
    """

    def __init__(self, gpu_id):
        if False:
            for i in range(10):
                print('nop')
        self.key = 0
        self.cudf_dataframe_dict = {}
        self.gpu_id = gpu_id

    def apply_non_persistent(self, first, other, func, **kwargs):
        if False:
            print('Hello World!')
        "\n        Apply `func` to values associated with `first`/`other` keys of `self.cudf_dataframe_dict`.\n\n        Parameters\n        ----------\n        first : int\n            The first key associated with dataframe from `self.cudf_dataframe_dict`.\n        other : int\n            The second key associated with dataframe from `self.cudf_dataframe_dict`.\n            If it isn't a real key, the `func` will be applied to the `first` only.\n        func : callable\n            A function to apply.\n        **kwargs : dict\n            Additional keywords arguments to be passed in `func`.\n\n        Returns\n        -------\n        The type of return of `func`\n            The result of the `func` (will be a ``ray.ObjectRef`` in outside level).\n        "
        df1 = self.cudf_dataframe_dict[first]
        df2 = self.cudf_dataframe_dict[other] if other else None
        if not df2:
            result = func(df1, **kwargs)
        else:
            result = func(df1, df2, **kwargs)
        return result

    def apply(self, first, other, func, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Apply `func` to values associated with `first`/`other` keys of `self.cudf_dataframe_dict` with storing of the result.\n\n        Store the return value of `func` (a new ``cudf.DataFrame``)\n        into `self.cudf_dataframe_dict`.\n\n        Parameters\n        ----------\n        first : int\n            The first key associated with dataframe from `self.cudf_dataframe_dict`.\n        other : int or ray.ObjectRef\n            The second key associated with dataframe from `self.cudf_dataframe_dict`.\n            If it isn't a real key, the `func` will be applied to the `first` only.\n        func : callable\n            A function to apply.\n        **kwargs : dict\n            Additional keywords arguments to be passed in `func`.\n\n        Returns\n        -------\n        int\n            The new key of the new dataframe stored in `self.cudf_dataframe_dict`\n            (will be a ``ray.ObjectRef`` in outside level).\n        "
        df1 = self.cudf_dataframe_dict[first]
        if not other:
            result = func(df1, **kwargs)
            return self.store_new_df(result)
        if not isinstance(other, int):
            assert isinstance(other, ray.ObjectRef)
            df2 = RayWrapper.materialize(other)
        else:
            df2 = self.cudf_dataframe_dict[other]
        result = func(df1, df2, **kwargs)
        return self.store_new_df(result)

    def reduce(self, first, others, func, axis=0, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Apply `func` to values associated with `first` key and `others` keys of `self.cudf_dataframe_dict` with storing of the result.\n\n        Dataframes associated with `others` keys will be concatenated to one\n        dataframe.\n\n        Store the return value of `func` (a new ``cudf.DataFrame``)\n        into `self.cudf_dataframe_dict`.\n\n        Parameters\n        ----------\n        first : int\n            The first key associated with dataframe from `self.cudf_dataframe_dict`.\n        others : list of int / list of ray.ObjectRef\n            The list of keys associated with dataframe from `self.cudf_dataframe_dict`.\n        func : callable\n            A function to apply.\n        axis : {0, 1}, default: 0\n            An axis corresponding to a particular row/column of the dataframe.\n        **kwargs : dict\n            Additional keywords arguments to be passed in `func`.\n\n        Returns\n        -------\n        int\n            The new key of the new dataframe stored in `self.cudf_dataframe_dict`\n            (will be a ``ray.ObjectRef`` in outside level).\n\n        Notes\n        -----\n        If ``len(others) == 0`` `func` should be able to work with 2nd\n        positional argument with None value.\n        '
        join_func = cudf.DataFrame.join if not axis else lambda x, y: cudf.concat([x, y])
        if not isinstance(others[0], int):
            other_dfs = RayWrapper.materialize(others)
        else:
            other_dfs = [self.cudf_dataframe_dict[i] for i in others]
        df1 = self.cudf_dataframe_dict[first]
        df2 = others[0] if len(others) >= 1 else None
        for i in range(1, len(others)):
            df2 = join_func(df2, other_dfs[i])
        result = func(df1, df2, **kwargs)
        return self.store_new_df(result)

    def store_new_df(self, df):
        if False:
            i = 10
            return i + 15
        '\n        Store `df` in `self.cudf_dataframe_dict`.\n\n        Parameters\n        ----------\n        df : cudf.DataFrame\n            The ``cudf.DataFrame`` to be added.\n\n        Returns\n        -------\n        int\n            The key associated with added dataframe\n            (will be a ``ray.ObjectRef`` in outside level).\n        '
        self.key += 1
        self.cudf_dataframe_dict[self.key] = df
        return self.key

    def free(self, key):
        if False:
            print('Hello World!')
        '\n        Free the dataFrame and associated `key` out of `self.cudf_dataframe_dict`.\n\n        Parameters\n        ----------\n        key : int\n            The key to be deleted.\n        '
        if key in self.cudf_dataframe_dict:
            del self.cudf_dataframe_dict[key]

    def get_id(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the `self.gpu_id` from this object.\n\n        Returns\n        -------\n        int\n            The gpu_id from this object\n            (will be a ``ray.ObjectRef`` in outside level).\n        '
        return self.gpu_id

    def get_oid(self, key):
        if False:
            while True:
                i = 10
        '\n        Get the value from `self.cudf_dataframe_dict` by `key`.\n\n        Parameters\n        ----------\n        key : int\n            The key to get value.\n\n        Returns\n        -------\n        cudf.DataFrame\n            Dataframe corresponding to `key`(will be a ``ray.ObjectRef``\n            in outside level).\n        '
        return self.cudf_dataframe_dict[key]

    def put(self, pandas_df):
        if False:
            while True:
                i = 10
        '\n        Convert `pandas_df` to ``cudf.DataFrame`` and put it to `self.cudf_dataframe_dict`.\n\n        Parameters\n        ----------\n        pandas_df : pandas.DataFrame/pandas.Series\n            A pandas DataFrame/Series to be added.\n\n        Returns\n        -------\n        int\n            The key associated with added dataframe\n            (will be a ``ray.ObjectRef`` in outside level).\n        '
        if isinstance(pandas_df, pandas.Series):
            pandas_df = pandas_df.to_frame()
        return self.store_new_df(cudf.from_pandas(pandas_df))