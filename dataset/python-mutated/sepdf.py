import pandas as pd
from typing import Dict, Iterable, Union

def align_index(df_dict, join):
    if False:
        while True:
            i = 10
    res = {}
    for (k, df) in df_dict.items():
        if join is not None and k != join:
            df = df.reindex(df_dict[join].index)
        res[k] = df
    return res

class SepDataFrame:
    """
    (Sep)erate DataFrame
    We usually concat multiple dataframe to be processed together(Such as feature, label, weight, filter).
    However, they are usually be used separately at last.
    This will result in extra cost for concatenating and splitting data(reshaping and copying data in the memory is very expensive)

    SepDataFrame tries to act like a DataFrame whose column with multiindex
    """

    def __init__(self, df_dict: Dict[str, pd.DataFrame], join: str, skip_align=False):
        if False:
            i = 10
            return i + 15
        '\n        initialize the data based on the dataframe dictionary\n\n        Parameters\n        ----------\n        df_dict : Dict[str, pd.DataFrame]\n            dataframe dictionary\n        join : str\n            how to join the data\n            It will reindex the dataframe based on the join key.\n            If join is None, the reindex step will be skipped\n\n        skip_align :\n            for some cases, we can improve performance by skipping aligning index\n        '
        self.join = join
        if skip_align:
            self._df_dict = df_dict
        else:
            self._df_dict = align_index(df_dict, join)

    @property
    def loc(self):
        if False:
            return 10
        return SDFLoc(self, join=self.join)

    @property
    def index(self):
        if False:
            print('Hello World!')
        return self._df_dict[self.join].index

    def apply_each(self, method: str, skip_align=True, *args, **kwargs):
        if False:
            return 10
        '\n        Assumptions:\n        - inplace methods will return None\n        '
        inplace = False
        df_dict = {}
        for (k, df) in self._df_dict.items():
            df_dict[k] = getattr(df, method)(*args, **kwargs)
            if df_dict[k] is None:
                inplace = True
        if not inplace:
            return SepDataFrame(df_dict=df_dict, join=self.join, skip_align=skip_align)

    def sort_index(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return self.apply_each('sort_index', True, *args, **kwargs)

    def copy(self, *args, **kwargs):
        if False:
            return 10
        return self.apply_each('copy', True, *args, **kwargs)

    def _update_join(self):
        if False:
            while True:
                i = 10
        if self.join not in self:
            if len(self._df_dict) > 0:
                self.join = next(iter(self._df_dict.keys()))
            else:
                self.join = None

    def __getitem__(self, item):
        if False:
            print('Hello World!')
        return self._df_dict[item]

    def __setitem__(self, item: str, df: Union[pd.DataFrame, pd.Series]):
        if False:
            return 10
        if not isinstance(item, tuple):
            self._df_dict[item] = df
        else:
            (_df_dict_key, *col_name) = item
            col_name = tuple(col_name)
            if _df_dict_key in self._df_dict:
                if len(col_name) == 1:
                    col_name = col_name[0]
                self._df_dict[_df_dict_key][col_name] = df
            elif isinstance(df, pd.Series):
                if len(col_name) == 1:
                    col_name = col_name[0]
                self._df_dict[_df_dict_key] = df.to_frame(col_name)
            else:
                df_copy = df.copy()
                df_copy.columns = pd.MultiIndex.from_tuples([(*col_name, *idx) for idx in df.columns.to_list()])
                self._df_dict[_df_dict_key] = df_copy

    def __delitem__(self, item: str):
        if False:
            print('Hello World!')
        del self._df_dict[item]
        self._update_join()

    def __contains__(self, item):
        if False:
            print('Hello World!')
        return item in self._df_dict

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._df_dict[self.join])

    def droplevel(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        raise NotImplementedError(f'Please implement the `droplevel` method')

    @property
    def columns(self):
        if False:
            print('Hello World!')
        dfs = []
        for (k, df) in self._df_dict.items():
            df = df.head(0)
            df.columns = pd.MultiIndex.from_product([[k], df.columns])
            dfs.append(df)
        return pd.concat(dfs, axis=1).columns

    @staticmethod
    def merge(df_dict: Dict[str, pd.DataFrame], join: str):
        if False:
            while True:
                i = 10
        all_df = df_dict[join]
        for (k, df) in df_dict.items():
            if k != join:
                all_df = all_df.join(df)
        return all_df

class SDFLoc:
    """Mock Class"""

    def __init__(self, sdf: SepDataFrame, join):
        if False:
            i = 10
            return i + 15
        self._sdf = sdf
        self.axis = None
        self.join = join

    def __call__(self, axis):
        if False:
            for i in range(10):
                print('nop')
        self.axis = axis
        return self

    def __getitem__(self, args):
        if False:
            for i in range(10):
                print('nop')
        if self.axis == 1:
            if isinstance(args, str):
                return self._sdf[args]
            elif isinstance(args, (tuple, list)):
                new_df_dict = {k: self._sdf[k] for k in args}
                return SepDataFrame(new_df_dict, join=self.join if self.join in args else args[0], skip_align=True)
            else:
                raise NotImplementedError(f'This type of input is not supported')
        elif self.axis == 0:
            return SepDataFrame({k: df.loc(axis=0)[args] for (k, df) in self._sdf._df_dict.items()}, join=self.join, skip_align=True)
        else:
            df = self._sdf
            if isinstance(args, tuple):
                (ax0, *ax1) = args
                if len(ax1) == 0:
                    ax1 = None
                if ax1 is not None:
                    df = df.loc(axis=1)[ax1]
                if ax0 is not None:
                    df = df.loc(axis=0)[ax0]
                return df
            else:
                return df.loc(axis=0)[args]
import builtins

def _isinstance(instance, cls):
    if False:
        i = 10
        return i + 15
    if isinstance_orig(instance, SepDataFrame):
        if isinstance(cls, Iterable):
            for c in cls:
                if c is pd.DataFrame:
                    return True
        elif cls is pd.DataFrame:
            return True
    return isinstance_orig(instance, cls)
builtins.isinstance_orig = builtins.isinstance
builtins.isinstance = _isinstance
if __name__ == '__main__':
    sdf = SepDataFrame({}, join=None)
    print(isinstance(sdf, (pd.DataFrame,)))
    print(isinstance(sdf, pd.DataFrame))