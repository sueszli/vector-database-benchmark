from ..doctools import document
from .stat import stat

@document
class stat_unique(stat):
    """
    Remove duplicates

    {usage}

    Parameters
    ----------
    {common_parameters}
    """
    DEFAULT_PARAMS = {'geom': 'point', 'position': 'identity', 'na_rm': False}

    @classmethod
    def compute_panel(cls, data, scales, **params):
        if False:
            while True:
                i = 10
        return data.drop_duplicates()