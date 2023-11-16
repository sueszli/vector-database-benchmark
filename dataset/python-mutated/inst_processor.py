import abc
import json
import pandas as pd

class InstProcessor:

    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame, instrument, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        process the data\n\n        NOTE: **The processor could change the content of `df` inplace !!!!! **\n        User should keep a copy of data outside\n\n        Parameters\n        ----------\n        df : pd.DataFrame\n            The raw_df of handler or result from previous processor.\n        '

    def __str__(self):
        if False:
            while True:
                i = 10
        return f'{self.__class__.__name__}:{json.dumps(self.__dict__, sort_keys=True, default=str)}'