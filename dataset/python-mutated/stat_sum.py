from ..doctools import document
from ..mapping.aes import ALL_AESTHETICS
from ..mapping.evaluation import after_stat
from ..utils import groupby_apply
from .stat import stat

@document
class stat_sum(stat):
    """
    Sum unique values

    Useful for overplotting on scatterplots.

    {usage}

    Parameters
    ----------
    {common_parameters}
    """
    _aesthetics_doc = "\n    {aesthetics_table}\n\n    .. rubric:: Options for computed aesthetics\n\n    ::\n\n        'n'     # Number of observations at a position\n        'prop'  # Ratio of points in that panel at a position\n\n    "
    REQUIRED_AES = {'x', 'y'}
    DEFAULT_PARAMS = {'geom': 'point', 'position': 'identity', 'na_rm': False}
    DEFAULT_AES = {'size': after_stat('n'), 'weight': 1}
    CREATES = {'n', 'prop'}

    @classmethod
    def compute_panel(cls, data, scales, **params):
        if False:
            print('Hello World!')
        if 'weight' not in data:
            data['weight'] = 1

        def count(df):
            if False:
                print('Hello World!')
            '\n            Do a weighted count\n            '
            df['n'] = df['weight'].sum()
            return df.iloc[0:1]

        def ave(df):
            if False:
                return 10
            '\n            Calculate proportion values\n            '
            df['prop'] = df['n'] / df['n'].sum()
            return df
        s: set[str] = set(data.columns) & ALL_AESTHETICS
        by = list(s.difference(['weight']))
        counts = groupby_apply(data, by, count)
        counts = groupby_apply(counts, 'group', ave)
        return counts