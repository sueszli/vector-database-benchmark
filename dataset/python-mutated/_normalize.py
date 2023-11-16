from __future__ import annotations
from collections import abc, defaultdict
import copy
from typing import TYPE_CHECKING, Any, DefaultDict
import numpy as np
from pandas._libs.writers import convert_json_to_lines
import pandas as pd
from pandas import DataFrame
if TYPE_CHECKING:
    from collections.abc import Iterable
    from pandas._typing import IgnoreRaise, Scalar

def convert_to_line_delimits(s: str) -> str:
    if False:
        return 10
    '\n    Helper function that converts JSON lists to line delimited JSON.\n    '
    if not s[0] == '[' and s[-1] == ']':
        return s
    s = s[1:-1]
    return convert_json_to_lines(s)

def nested_to_record(ds, prefix: str='', sep: str='.', level: int=0, max_level: int | None=None):
    if False:
        print('Hello World!')
    '\n    A simplified json_normalize\n\n    Converts a nested dict into a flat dict ("record"), unlike json_normalize,\n    it does not attempt to extract a subset of the data.\n\n    Parameters\n    ----------\n    ds : dict or list of dicts\n    prefix: the prefix, optional, default: ""\n    sep : str, default \'.\'\n        Nested records will generate names separated by sep,\n        e.g., for sep=\'.\', { \'foo\' : { \'bar\' : 0 } } -> foo.bar\n    level: int, optional, default: 0\n        The number of levels in the json string.\n\n    max_level: int, optional, default: None\n        The max depth to normalize.\n\n    Returns\n    -------\n    d - dict or list of dicts, matching `ds`\n\n    Examples\n    --------\n    >>> nested_to_record(\n    ...     dict(flat1=1, dict1=dict(c=1, d=2), nested=dict(e=dict(c=1, d=2), d=2))\n    ... )\n    {\'flat1\': 1, \'dict1.c\': 1, \'dict1.d\': 2, \'nested.e.c\': 1, \'nested.e.d\': 2, \'nested.d\': 2}\n    '
    singleton = False
    if isinstance(ds, dict):
        ds = [ds]
        singleton = True
    new_ds = []
    for d in ds:
        new_d = copy.deepcopy(d)
        for (k, v) in d.items():
            if not isinstance(k, str):
                k = str(k)
            if level == 0:
                newkey = k
            else:
                newkey = prefix + sep + k
            if not isinstance(v, dict) or (max_level is not None and level >= max_level):
                if level != 0:
                    v = new_d.pop(k)
                    new_d[newkey] = v
                continue
            v = new_d.pop(k)
            new_d.update(nested_to_record(v, newkey, sep, level + 1, max_level))
        new_ds.append(new_d)
    if singleton:
        return new_ds[0]
    return new_ds

def _normalise_json(data: Any, key_string: str, normalized_dict: dict[str, Any], separator: str) -> dict[str, Any]:
    if False:
        while True:
            i = 10
    "\n    Main recursive function\n    Designed for the most basic use case of pd.json_normalize(data)\n    intended as a performance improvement, see #15621\n\n    Parameters\n    ----------\n    data : Any\n        Type dependent on types contained within nested Json\n    key_string : str\n        New key (with separator(s) in) for data\n    normalized_dict : dict\n        The new normalized/flattened Json dict\n    separator : str, default '.'\n        Nested records will generate names separated by sep,\n        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar\n    "
    if isinstance(data, dict):
        for (key, value) in data.items():
            new_key = f'{key_string}{separator}{key}'
            if not key_string:
                new_key = new_key.removeprefix(separator)
            _normalise_json(data=value, key_string=new_key, normalized_dict=normalized_dict, separator=separator)
    else:
        normalized_dict[key_string] = data
    return normalized_dict

def _normalise_json_ordered(data: dict[str, Any], separator: str) -> dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Order the top level keys and then recursively go to depth\n\n    Parameters\n    ----------\n    data : dict or list of dicts\n    separator : str, default '.'\n        Nested records will generate names separated by sep,\n        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar\n\n    Returns\n    -------\n    dict or list of dicts, matching `normalised_json_object`\n    "
    top_dict_ = {k: v for (k, v) in data.items() if not isinstance(v, dict)}
    nested_dict_ = _normalise_json(data={k: v for (k, v) in data.items() if isinstance(v, dict)}, key_string='', normalized_dict={}, separator=separator)
    return {**top_dict_, **nested_dict_}

def _simple_json_normalize(ds: dict | list[dict], sep: str='.') -> dict | list[dict] | Any:
    if False:
        print('Hello World!')
    '\n    A optimized basic json_normalize\n\n    Converts a nested dict into a flat dict ("record"), unlike\n    json_normalize and nested_to_record it doesn\'t do anything clever.\n    But for the most basic use cases it enhances performance.\n    E.g. pd.json_normalize(data)\n\n    Parameters\n    ----------\n    ds : dict or list of dicts\n    sep : str, default \'.\'\n        Nested records will generate names separated by sep,\n        e.g., for sep=\'.\', { \'foo\' : { \'bar\' : 0 } } -> foo.bar\n\n    Returns\n    -------\n    frame : DataFrame\n    d - dict or list of dicts, matching `normalised_json_object`\n\n    Examples\n    --------\n    >>> _simple_json_normalize(\n    ...     {\n    ...         "flat1": 1,\n    ...         "dict1": {"c": 1, "d": 2},\n    ...         "nested": {"e": {"c": 1, "d": 2}, "d": 2},\n    ...     }\n    ... )\n    {\'flat1\': 1, \'dict1.c\': 1, \'dict1.d\': 2, \'nested.e.c\': 1, \'nested.e.d\': 2, \'nested.d\': 2}\n\n    '
    normalised_json_object = {}
    if isinstance(ds, dict):
        normalised_json_object = _normalise_json_ordered(data=ds, separator=sep)
    elif isinstance(ds, list):
        normalised_json_list = [_simple_json_normalize(row, sep=sep) for row in ds]
        return normalised_json_list
    return normalised_json_object

def json_normalize(data: dict | list[dict], record_path: str | list | None=None, meta: str | list[str | list[str]] | None=None, meta_prefix: str | None=None, record_prefix: str | None=None, errors: IgnoreRaise='raise', sep: str='.', max_level: int | None=None) -> DataFrame:
    if False:
        print('Hello World!')
    '\n    Normalize semi-structured JSON data into a flat table.\n\n    Parameters\n    ----------\n    data : dict or list of dicts\n        Unserialized JSON objects.\n    record_path : str or list of str, default None\n        Path in each object to list of records. If not passed, data will be\n        assumed to be an array of records.\n    meta : list of paths (str or list of str), default None\n        Fields to use as metadata for each record in resulting table.\n    meta_prefix : str, default None\n        If True, prefix records with dotted (?) path, e.g. foo.bar.field if\n        meta is [\'foo\', \'bar\'].\n    record_prefix : str, default None\n        If True, prefix records with dotted (?) path, e.g. foo.bar.field if\n        path to records is [\'foo\', \'bar\'].\n    errors : {\'raise\', \'ignore\'}, default \'raise\'\n        Configures error handling.\n\n        * \'ignore\' : will ignore KeyError if keys listed in meta are not\n          always present.\n        * \'raise\' : will raise KeyError if keys listed in meta are not\n          always present.\n    sep : str, default \'.\'\n        Nested records will generate names separated by sep.\n        e.g., for sep=\'.\', {\'foo\': {\'bar\': 0}} -> foo.bar.\n    max_level : int, default None\n        Max number of levels(depth of dict) to normalize.\n        if None, normalizes all levels.\n\n    Returns\n    -------\n    frame : DataFrame\n    Normalize semi-structured JSON data into a flat table.\n\n    Examples\n    --------\n    >>> data = [\n    ...     {"id": 1, "name": {"first": "Coleen", "last": "Volk"}},\n    ...     {"name": {"given": "Mark", "family": "Regner"}},\n    ...     {"id": 2, "name": "Faye Raker"},\n    ... ]\n    >>> pd.json_normalize(data)\n        id name.first name.last name.given name.family        name\n    0  1.0     Coleen      Volk        NaN         NaN         NaN\n    1  NaN        NaN       NaN       Mark      Regner         NaN\n    2  2.0        NaN       NaN        NaN         NaN  Faye Raker\n\n    >>> data = [\n    ...     {\n    ...         "id": 1,\n    ...         "name": "Cole Volk",\n    ...         "fitness": {"height": 130, "weight": 60},\n    ...     },\n    ...     {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},\n    ...     {\n    ...         "id": 2,\n    ...         "name": "Faye Raker",\n    ...         "fitness": {"height": 130, "weight": 60},\n    ...     },\n    ... ]\n    >>> pd.json_normalize(data, max_level=0)\n        id        name                        fitness\n    0  1.0   Cole Volk  {\'height\': 130, \'weight\': 60}\n    1  NaN    Mark Reg  {\'height\': 130, \'weight\': 60}\n    2  2.0  Faye Raker  {\'height\': 130, \'weight\': 60}\n\n    Normalizes nested data up to level 1.\n\n    >>> data = [\n    ...     {\n    ...         "id": 1,\n    ...         "name": "Cole Volk",\n    ...         "fitness": {"height": 130, "weight": 60},\n    ...     },\n    ...     {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},\n    ...     {\n    ...         "id": 2,\n    ...         "name": "Faye Raker",\n    ...         "fitness": {"height": 130, "weight": 60},\n    ...     },\n    ... ]\n    >>> pd.json_normalize(data, max_level=1)\n        id        name  fitness.height  fitness.weight\n    0  1.0   Cole Volk             130              60\n    1  NaN    Mark Reg             130              60\n    2  2.0  Faye Raker             130              60\n\n    >>> data = [\n    ...     {\n    ...         "state": "Florida",\n    ...         "shortname": "FL",\n    ...         "info": {"governor": "Rick Scott"},\n    ...         "counties": [\n    ...             {"name": "Dade", "population": 12345},\n    ...             {"name": "Broward", "population": 40000},\n    ...             {"name": "Palm Beach", "population": 60000},\n    ...         ],\n    ...     },\n    ...     {\n    ...         "state": "Ohio",\n    ...         "shortname": "OH",\n    ...         "info": {"governor": "John Kasich"},\n    ...         "counties": [\n    ...             {"name": "Summit", "population": 1234},\n    ...             {"name": "Cuyahoga", "population": 1337},\n    ...         ],\n    ...     },\n    ... ]\n    >>> result = pd.json_normalize(\n    ...     data, "counties", ["state", "shortname", ["info", "governor"]]\n    ... )\n    >>> result\n             name  population    state shortname info.governor\n    0        Dade       12345   Florida    FL    Rick Scott\n    1     Broward       40000   Florida    FL    Rick Scott\n    2  Palm Beach       60000   Florida    FL    Rick Scott\n    3      Summit        1234   Ohio       OH    John Kasich\n    4    Cuyahoga        1337   Ohio       OH    John Kasich\n\n    >>> data = {"A": [1, 2]}\n    >>> pd.json_normalize(data, "A", record_prefix="Prefix.")\n        Prefix.0\n    0          1\n    1          2\n\n    Returns normalized data with columns prefixed with the given string.\n    '

    def _pull_field(js: dict[str, Any], spec: list | str, extract_record: bool=False) -> Scalar | Iterable:
        if False:
            while True:
                i = 10
        'Internal function to pull field'
        result = js
        try:
            if isinstance(spec, list):
                for field in spec:
                    if result is None:
                        raise KeyError(field)
                    result = result[field]
            else:
                result = result[spec]
        except KeyError as e:
            if extract_record:
                raise KeyError(f'Key {e} not found. If specifying a record_path, all elements of data should have the path.') from e
            if errors == 'ignore':
                return np.nan
            else:
                raise KeyError(f"Key {e} not found. To replace missing values of {e} with np.nan, pass in errors='ignore'") from e
        return result

    def _pull_records(js: dict[str, Any], spec: list | str) -> list:
        if False:
            print('Hello World!')
        '\n        Internal function to pull field for records, and similar to\n        _pull_field, but require to return list. And will raise error\n        if has non iterable value.\n        '
        result = _pull_field(js, spec, extract_record=True)
        if not isinstance(result, list):
            if pd.isnull(result):
                result = []
            else:
                raise TypeError(f'{js} has non list value {result} for path {spec}. Must be list or null.')
        return result
    if isinstance(data, list) and (not data):
        return DataFrame()
    elif isinstance(data, dict):
        data = [data]
    elif isinstance(data, abc.Iterable) and (not isinstance(data, str)):
        data = list(data)
    else:
        raise NotImplementedError
    if record_path is None and meta is None and (meta_prefix is None) and (record_prefix is None) and (max_level is None):
        return DataFrame(_simple_json_normalize(data, sep=sep))
    if record_path is None:
        if any(([isinstance(x, dict) for x in y.values()] for y in data)):
            data = nested_to_record(data, sep=sep, max_level=max_level)
        return DataFrame(data)
    elif not isinstance(record_path, list):
        record_path = [record_path]
    if meta is None:
        meta = []
    elif not isinstance(meta, list):
        meta = [meta]
    _meta = [m if isinstance(m, list) else [m] for m in meta]
    records: list = []
    lengths = []
    meta_vals: DefaultDict = defaultdict(list)
    meta_keys = [sep.join(val) for val in _meta]

    def _recursive_extract(data, path, seen_meta, level: int=0) -> None:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(data, dict):
            data = [data]
        if len(path) > 1:
            for obj in data:
                for (val, key) in zip(_meta, meta_keys):
                    if level + 1 == len(val):
                        seen_meta[key] = _pull_field(obj, val[-1])
                _recursive_extract(obj[path[0]], path[1:], seen_meta, level=level + 1)
        else:
            for obj in data:
                recs = _pull_records(obj, path[0])
                recs = [nested_to_record(r, sep=sep, max_level=max_level) if isinstance(r, dict) else r for r in recs]
                lengths.append(len(recs))
                for (val, key) in zip(_meta, meta_keys):
                    if level + 1 > len(val):
                        meta_val = seen_meta[key]
                    else:
                        meta_val = _pull_field(obj, val[level:])
                    meta_vals[key].append(meta_val)
                records.extend(recs)
    _recursive_extract(data, record_path, {}, level=0)
    result = DataFrame(records)
    if record_prefix is not None:
        result = result.rename(columns=lambda x: f'{record_prefix}{x}')
    for (k, v) in meta_vals.items():
        if meta_prefix is not None:
            k = meta_prefix + k
        if k in result:
            raise ValueError(f'Conflicting metadata name {k}, need distinguishing prefix ')
        values = np.array(v, dtype=object)
        if values.ndim > 1:
            values = np.empty((len(v),), dtype=object)
            for (i, v) in enumerate(v):
                values[i] = v
        result[k] = values.repeat(lengths)
    return result