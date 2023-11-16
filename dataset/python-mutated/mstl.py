__all__ = ['mstl']
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import statsmodels.api as sm

def mstl(x: np.ndarray, period: Union[int, List[int]], blambda: Optional[float]=None, iterate: int=1, s_window: Optional[np.ndarray]=None, stl_kwargs: Optional[Dict]=dict()):
    if False:
        while True:
            i = 10
    if s_window is None:
        s_window = 7 + 4 * np.arange(1, 7)
    origx = x
    n = len(x)
    msts = [period] if isinstance(period, int) else period
    iterate = 1
    if x.ndim == 2:
        x = x[:, 0]
    if np.isnan(x).any():
        raise Exception('`mstl` cannot handle missing values. Please raise an issue to include this feature.')
    if blambda is not None:
        raise Exception('`blambda` not implemented yet. Please rise an issue to include this feature.')
    if msts[0] > 1:
        seas = np.zeros((len(msts), n))
        deseas = np.copy(x)
        if len(s_window) == 1:
            s_window = np.repeat(s_window, len(msts))
        for j in range(iterate):
            for (i, seas_) in enumerate(msts, start=0):
                deseas = deseas + seas[i]
                fit = sm.tsa.STL(deseas, period=seas_, seasonal=s_window[i], **stl_kwargs).fit()
                seas[i] = fit.seasonal
                deseas = deseas - seas[i]
        trend = fit.trend
    else:
        try:
            from supersmoother import SuperSmoother
        except ImportError as e:
            print('supersmoother is required for mstl with period=1')
            raise e
        deseas = x
        t = 1 + np.arange(n)
        trend = SuperSmoother().fit(t, x).predict(t)
    deseas[np.isnan(origx)] = np.nan
    remainder = deseas - trend
    output = {'data': origx, 'trend': trend}
    if msts is not None and msts[0] > 1:
        if len(msts) == 1:
            output['seasonal'] = seas[0]
        else:
            for (i, seas_) in enumerate(msts, start=0):
                output[f'seasonal{seas_}'] = seas[i]
    output['remainder'] = remainder
    return pd.DataFrame(output)