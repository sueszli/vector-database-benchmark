from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd

def _frequency_table(freqtable: pd.Series, n: int, max_number_to_print: int) -> List[Dict[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    if max_number_to_print > n:
        max_number_to_print = n
    if max_number_to_print < len(freqtable):
        freq_other = np.sum(freqtable.iloc[max_number_to_print:])
        min_freq = freqtable.values[max_number_to_print]
    else:
        freq_other = 0
        min_freq = 0
    freq_missing = n - np.sum(freqtable)
    if len(freqtable) == 0:
        return []
    max_freq = max(freqtable.values[0], freq_other, freq_missing)
    if max_freq == 0:
        return []
    rows = []
    for (label, freq) in freqtable.iloc[0:max_number_to_print].items():
        rows.append({'label': label, 'width': freq / max_freq, 'count': freq, 'percentage': float(freq) / n, 'n': n, 'extra_class': ''})
    if freq_other > min_freq:
        other_count = str(freqtable.count() - max_number_to_print)
        rows.append({'label': f'Other values ({other_count})', 'width': freq_other / max_freq, 'count': freq_other, 'percentage': min(float(freq_other) / n, 1.0), 'n': n, 'extra_class': 'other'})
    if freq_missing > min_freq:
        rows.append({'label': '(Missing)', 'width': freq_missing / max_freq, 'count': freq_missing, 'percentage': float(freq_missing) / n, 'n': n, 'extra_class': 'missing'})
    return rows

def freq_table(freqtable: Union[pd.Series, List[pd.Series]], n: Union[int, List[int]], max_number_to_print: int) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    if False:
        for i in range(10):
            print('nop')
    'Render the rows for a frequency table (value, count).\n\n    Args:\n      freqtable: The frequency table.\n      n: The total number of values.\n      max_number_to_print: The maximum number of observations to print.\n\n    Returns:\n        The rows of the frequency table.\n    '
    if isinstance(freqtable, list) and isinstance(n, list):
        return [_frequency_table(v, n2, max_number_to_print) for (v, n2) in zip(freqtable, n)]
    else:
        return [_frequency_table(freqtable, n, max_number_to_print)]

def _extreme_obs_table(freqtable: pd.Series, number_to_print: int, n: int) -> List[Dict[str, Any]]:
    if False:
        while True:
            i = 10
    obs_to_print = freqtable.iloc[:number_to_print]
    max_freq = obs_to_print.max()
    rows = [{'label': label, 'width': freq / max_freq if max_freq != 0 else 0, 'count': freq, 'percentage': float(freq) / n, 'extra_class': '', 'n': n} for (label, freq) in obs_to_print.items()]
    return rows

def extreme_obs_table(freqtable: Union[pd.Series, List[pd.Series]], number_to_print: int, n: Union[int, List[int]]) -> List[List[Dict[str, Any]]]:
    if False:
        return 10
    'Similar to the frequency table, for extreme observations.\n\n    Args:\n      freqtable: The (sorted) frequency table.\n      number_to_print: The number of observations to print.\n      n: The total number of observations.\n\n    Returns:\n        The HTML rendering of the extreme observation table.\n    '
    if isinstance(freqtable, list) and isinstance(n, list):
        return [_extreme_obs_table(v, number_to_print, n1) for (v, n1) in zip(freqtable, n)]
    return [_extreme_obs_table(freqtable, number_to_print, n)]