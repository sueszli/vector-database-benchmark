"""
Moving Average
--------------
"""
from darts.models.filtering.filtering_model import FilteringModel
from darts.timeseries import TimeSeries

class MovingAverageFilter(FilteringModel):
    """
    A simple moving average filter. Works on deterministic and stochastic series.
    """

    def __init__(self, window: int, centered: bool=True):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        window\n            The length of the window over which to average values\n        centered\n            Set the labels at the center of the window. If not set, the averaged values are lagging after\n            the original values.\n        '
        super().__init__()
        self.window = window
        self.centered = centered

    def filter(self, series: TimeSeries):
        if False:
            return 10
        "\n        Computes a moving average of this series' values and returns a new TimeSeries.\n        The returned series has the same length and time axis as `series`. (Note that this might create border effects).\n\n        Parameters\n        ----------\n        series\n            The a deterministic series to average\n\n        Returns\n        -------\n        TimeSeries\n            A time series containing the average values\n        "
        transformation = {'function': 'mean', 'mode': 'rolling', 'window': self.window, 'center': self.centered, 'min_periods': 1}
        return series.window_transform(transforms=transformation, forecasting_safe=False)