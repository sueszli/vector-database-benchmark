from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
from turicreate.toolkits._internal_utils import _raise_error_if_not_sframe, _raise_error_if_not_sarray

def stack_annotations(annotations_sarray):
    if False:
        print('Hello World!')
    '\n    Converts object detection annotations (ground truth or predictions) to\n    stacked format (an `SFrame` where each row is one object instance).\n\n    Parameters\n    ----------\n    annotations_sarray: SArray\n        An `SArray` with unstacked predictions, exactly formatted as the\n        annotations column when training an object detector or when making\n        predictions.\n\n    Returns\n    -------\n    annotations_sframe: An `SFrame` with stacked annotations.\n\n    See also\n    --------\n    unstack_annotations\n\n    Examples\n    --------\n    Predictions are returned by the object detector in unstacked format:\n\n    >>> predictions = detector.predict(images)\n\n    By converting it to stacked format, it is easier to get an overview of\n    object instances:\n\n    >>> turicreate.object_detector.util.stack_annotations(predictions)\n    Data:\n    +--------+------------+-------+-------+-------+-------+--------+\n    | row_id | confidence | label |   x   |   y   | width | height |\n    +--------+------------+-------+-------+-------+-------+--------+\n    |   0    |    0.98    |  dog  | 123.0 | 128.0 |  80.0 | 182.0  |\n    |   0    |    0.67    |  cat  | 150.0 | 183.0 | 129.0 | 101.0  |\n    |   1    |    0.8     |  dog  |  50.0 | 432.0 |  65.0 |  98.0  |\n    +--------+------------+-------+-------+-------+-------+--------+\n    [3 rows x 7 columns]\n    '
    _raise_error_if_not_sarray(annotations_sarray, variable_name='annotations_sarray')
    sf = _tc.SFrame({'annotations': annotations_sarray}).add_row_number('row_id')
    sf = sf.stack('annotations', new_column_name='annotations', drop_na=True)
    if len(sf) == 0:
        cols = ['row_id', 'confidence', 'label', 'height', 'width', 'x', 'y']
        return _tc.SFrame({k: [] for k in cols})
    sf = sf.unpack('annotations', column_name_prefix='')
    sf = sf.unpack('coordinates', column_name_prefix='')
    del sf['type']
    return sf

def unstack_annotations(annotations_sframe, num_rows=None):
    if False:
        while True:
            i = 10
    "\n    Converts object detection annotations (ground truth or predictions) to\n    unstacked format (an `SArray` where each element is a list of object\n    instances).\n\n    Parameters\n    ----------\n    annotations_sframe: SFrame\n        An `SFrame` with stacked predictions, produced by the\n        `stack_annotations` function.\n\n    num_rows: int\n        Optionally specify the number of rows in your original dataset, so that\n        all get represented in the unstacked format, regardless of whether or\n        not they had instances or not.\n\n    Returns\n    -------\n    annotations_sarray: An `SArray` with unstacked annotations.\n\n    See also\n    --------\n    stack_annotations\n\n    Examples\n    --------\n    If you have annotations in stacked format:\n\n    >>> stacked_predictions\n    Data:\n    +--------+------------+-------+-------+-------+-------+--------+\n    | row_id | confidence | label |   x   |   y   | width | height |\n    +--------+------------+-------+-------+-------+-------+--------+\n    |   0    |    0.98    |  dog  | 123.0 | 128.0 |  80.0 | 182.0  |\n    |   0    |    0.67    |  cat  | 150.0 | 183.0 | 129.0 | 101.0  |\n    |   1    |    0.8     |  dog  |  50.0 | 432.0 |  65.0 |  98.0  |\n    +--------+------------+-------+-------+-------+-------+--------+\n    [3 rows x 7 columns]\n\n    They can be converted to unstacked format using this function:\n\n    >>> turicreate.object_detector.util.unstack_annotations(stacked_predictions)[0]\n    [{'confidence': 0.98,\n      'coordinates': {'height': 182.0, 'width': 80.0, 'x': 123.0, 'y': 128.0},\n      'label': 'dog',\n      'type': 'rectangle'},\n     {'confidence': 0.67,\n      'coordinates': {'height': 101.0, 'width': 129.0, 'x': 150.0, 'y': 183.0},\n      'label': 'cat',\n      'type': 'rectangle'}]\n    "
    _raise_error_if_not_sframe(annotations_sframe, variable_name='annotations_sframe')
    cols = ['label', 'type', 'coordinates']
    has_confidence = 'confidence' in annotations_sframe.column_names()
    if has_confidence:
        cols.append('confidence')
    if num_rows is None:
        if len(annotations_sframe) == 0:
            num_rows = 0
        else:
            num_rows = annotations_sframe['row_id'].max() + 1
    sf = annotations_sframe
    sf['type'] = 'rectangle'
    sf = sf.pack_columns(['x', 'y', 'width', 'height'], dtype=dict, new_column_name='coordinates')
    sf = sf.pack_columns(cols, dtype=dict, new_column_name='ann')
    sf = sf.unstack('ann', new_column_name='annotations')
    sf_all_ids = _tc.SFrame({'row_id': range(num_rows)})
    sf = sf.join(sf_all_ids, on='row_id', how='right')
    sf = sf.fillna('annotations', [])
    sf = sf.sort('row_id')
    annotations_sarray = sf['annotations']
    if has_confidence:
        annotations_sarray = annotations_sarray.apply(lambda x: sorted(x, key=lambda ann: ann['confidence'], reverse=True), dtype=list)
    return annotations_sarray