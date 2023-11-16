import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from funcy import first, last
from dvc.exceptions import DvcException
from dvc.render import FILENAME_FIELD, INDEX_FIELD, VERSION_FIELD
from . import Converter

class FieldNotFoundError(DvcException):

    def __init__(self, expected_field, found_fields):
        if False:
            return 10
        found_str = ', '.join(found_fields)
        super().__init__(f"Could not find provided field ('{expected_field}') in data fields ('{found_str}').")

def _lists(blob: Union[Dict, List]) -> Iterable[List]:
    if False:
        i = 10
        return i + 15
    if isinstance(blob, list):
        yield blob
    else:
        for (_, value) in blob.items():
            if isinstance(value, dict):
                yield from _lists(value)
            elif isinstance(value, list):
                yield value

def _file_field(*args):
    if False:
        i = 10
        return i + 15
    for axis_def in args:
        if axis_def is not None:
            for (file, val) in axis_def.items():
                if isinstance(val, str):
                    yield (file, val)
                elif isinstance(val, list):
                    for field in val:
                        yield (file, field)

def _find(filename: str, field: str, data_series: List[Tuple[str, str, Any]]):
    if False:
        print('Hello World!')
    for (data_file, data_field, data) in data_series:
        if data_file == filename and data_field == field:
            return (data_file, data_field, data)
    return None

def _verify_field(file2datapoints: Dict[str, List], filename: str, field: str):
    if False:
        for i in range(10):
            print('nop')
    if filename in file2datapoints:
        datapoint = first(file2datapoints[filename])
        if field not in datapoint:
            raise FieldNotFoundError(field, datapoint.keys())

def _get_xs(properties: Dict, file2datapoints: Dict[str, List[Dict]]):
    if False:
        i = 10
        return i + 15
    x = properties.get('x', None)
    if x is not None and isinstance(x, dict):
        for (filename, field) in _file_field(x):
            _verify_field(file2datapoints, filename, field)
            yield (filename, field)

def _get_ys(properties, file2datapoints: Dict[str, List[Dict]]):
    if False:
        print('Hello World!')
    y = properties.get('y', None)
    if y is not None:
        for (filename, field) in _file_field(y):
            _verify_field(file2datapoints, filename, field)
            yield (filename, field)

def _is_datapoints(lst: List[Dict]):
    if False:
        for i in range(10):
            print('nop')
    '\n    check if dict keys match, datapoints with different keys mgiht lead\n    to unexpected behavior\n    '
    return all((isinstance(item, dict) for item in lst)) and set(first(lst).keys()) == {key for keys in lst for key in keys}

def get_datapoints(file_content: Dict):
    if False:
        print('Hello World!')
    result: List[Dict[str, Any]] = []
    for lst in _lists(file_content):
        if _is_datapoints(lst):
            for (index, datapoint) in enumerate(lst):
                if len(result) <= index:
                    result.append({})
                result[index].update(datapoint)
    return result

class VegaConverter(Converter):
    """
    Class that takes care of converting unspecified data blob
    (Dict or List[Dict]) into datapoints (List[Dict]).
    If some properties that are required by Template class are missing
    ('x', 'y') it will attempt to fill in the blanks.
    """

    def __init__(self, plot_id: str, data: Optional[Dict]=None, properties: Optional[Dict]=None):
        if False:
            return 10
        super().__init__(plot_id, data, properties)
        self.plot_id = plot_id
        self.inferred_properties: Dict = {}

    def _infer_y_from_data(self):
        if False:
            return 10
        if self.plot_id in self.data:
            for lst in _lists(self.data[self.plot_id]):
                if all((isinstance(item, dict) for item in lst)):
                    datapoint = first(lst)
                    field = last(datapoint.keys())
                    self.inferred_properties['y'] = {self.plot_id: field}
                    break

    def _infer_x_y(self):
        if False:
            i = 10
            return i + 15
        x = self.properties.get('x', None)
        y = self.properties.get('y', None)
        if isinstance(x, str):
            self.inferred_properties['x'] = {}
            if isinstance(y, dict):
                for (file, fields) in y.items():
                    if isinstance(fields, list):
                        self.inferred_properties['x'][file] = [x] * len(fields)
                    else:
                        self.inferred_properties['x'][file] = x
            else:
                self.inferred_properties['x'][self.plot_id] = x
        if y is None:
            self._infer_y_from_data()
        elif not isinstance(y, dict):
            self.inferred_properties['y'] = {self.plot_id: y}

    def _find_datapoints(self):
        if False:
            print('Hello World!')
        result = {}
        for (file, content) in self.data.items():
            result[file] = get_datapoints(content)
        return result

    @staticmethod
    def infer_y_label(properties):
        if False:
            for i in range(10):
                print('nop')
        y_label = properties.get('y_label', None)
        if y_label is not None:
            return y_label
        y = properties.get('y', None)
        if isinstance(y, str):
            return y
        if isinstance(y, list):
            return 'y'
        if not isinstance(y, dict):
            return
        fields = {field for (_, field) in _file_field(y)}
        if len(fields) == 1:
            return first(fields)
        return 'y'

    @staticmethod
    def infer_x_label(properties):
        if False:
            for i in range(10):
                print('nop')
        x_label = properties.get('x_label', None)
        if x_label is not None:
            return x_label
        x = properties.get('x', None)
        if not isinstance(x, dict):
            return INDEX_FIELD
        fields = {field for (_, field) in _file_field(x)}
        if len(fields) == 1:
            return first(fields)
        return 'x'

    def flat_datapoints(self, revision):
        if False:
            print('Hello World!')
        (file2datapoints, properties) = self.convert()
        props_update = {}
        xs = list(_get_xs(properties, file2datapoints))
        if not xs:
            (x_file, x_field) = (None, INDEX_FIELD)
        else:
            (x_file, x_field) = xs[0]
        props_update['x'] = x_field
        ys = list(_get_ys(properties, file2datapoints))
        num_xs = len(xs)
        num_ys = len(ys)
        if num_xs > 1 and num_xs != num_ys:
            raise DvcException(f'Cannot have different number of x and y data sources. Found {num_xs} x and {num_ys} y data sources.')
        all_datapoints = []
        if ys:
            (_all_y_files, _all_y_fields) = list(zip(*ys))
            all_y_fields = set(_all_y_fields)
            all_y_files = set(_all_y_files)
        else:
            all_y_files = set()
            all_y_fields = set()
        if len(all_y_fields) > 1:
            props_update['y'] = 'dvc_inferred_y_value'
        else:
            props_update['y'] = first(all_y_fields)
        if len(all_y_files) > 1:
            common_prefix_len = len(os.path.commonpath(list(all_y_files)))
        else:
            common_prefix_len = 0
        for (i, (y_file, y_field)) in enumerate(ys):
            if num_xs > 1:
                (x_file, x_field) = xs[i]
            datapoints = [{**d} for d in file2datapoints.get(y_file, [])]
            if props_update.get('y', None) == 'dvc_inferred_y_value':
                _update_from_field(datapoints, field='dvc_inferred_y_value', source_field=y_field)
            if x_field == INDEX_FIELD and x_file is None:
                _update_from_index(datapoints, INDEX_FIELD)
            else:
                x_datapoints = file2datapoints.get(x_file, [])
                try:
                    _update_from_field(datapoints, field=x_field, source_datapoints=x_datapoints)
                except IndexError:
                    raise DvcException(f"Cannot join '{x_field}' from '{x_file}' and '{y_field}' from '{y_file}'. They have to have same length.")
            y_file_short = y_file[common_prefix_len:].strip('/\\')
            _update_all(datapoints, update_dict={VERSION_FIELD: {'revision': revision, FILENAME_FIELD: y_file_short, 'field': y_field}})
            all_datapoints.extend(datapoints)
        if not all_datapoints:
            return ([], {})
        properties = {**properties, **props_update}
        return (all_datapoints, properties)

    def convert(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Convert the data. Fill necessary fields ('x', 'y') and return both\n        generated datapoints and updated properties. `x`, `y` values and labels\n        are inferred and always provided.\n        "
        self._infer_x_y()
        datapoints = self._find_datapoints()
        properties = {**self.properties, **self.inferred_properties}
        properties['y_label'] = self.infer_y_label(properties)
        properties['x_label'] = self.infer_x_label(properties)
        return (datapoints, properties)

def _update_from_field(target_datapoints: List[Dict], field: str, source_datapoints: Optional[List[Dict]]=None, source_field: Optional[str]=None):
    if False:
        while True:
            i = 10
    if source_datapoints is None:
        source_datapoints = target_datapoints
    if source_field is None:
        source_field = field
    if len(source_datapoints) != len(target_datapoints):
        raise IndexError('Source and target datapoints must have the same length')
    for (index, datapoint) in enumerate(target_datapoints):
        source_datapoint = source_datapoints[index]
        if source_field in source_datapoint:
            datapoint[field] = source_datapoint[source_field]

def _update_from_index(datapoints: List[Dict], new_field: str):
    if False:
        print('Hello World!')
    for (index, datapoint) in enumerate(datapoints):
        datapoint[new_field] = index

def _update_all(datapoints: List[Dict], update_dict: Dict):
    if False:
        print('Hello World!')
    for datapoint in datapoints:
        datapoint.update(update_dict)