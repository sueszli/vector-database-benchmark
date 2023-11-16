from collections import defaultdict
import numpy as np

def array_to_binary(ar, obj=None, force_contiguous=True):
    if False:
        while True:
            i = 10
    if ar is None:
        return None
    if ar.dtype.kind not in ['u', 'i', 'f']:
        raise ValueError('unsupported dtype: %s' % ar.dtype)
    if ar.dtype == np.float64:
        ar = ar.astype(np.float32)
    if ar.dtype == np.int64:
        ar = ar.astype(np.int32)
    if force_contiguous and (not ar.flags['C_CONTIGUOUS']):
        ar = np.ascontiguousarray(ar)
    return {'value': memoryview(ar), 'dtype': str(ar.dtype), 'length': ar.shape[0], 'size': 1 if len(ar.shape) == 1 else ar.shape[1]}

def serialize_columns(data_set_cols, obj=None):
    if False:
        while True:
            i = 10
    if data_set_cols is None:
        return None
    layers = defaultdict(dict)
    length = {}
    for col in data_set_cols:
        accessor_attribute = array_to_binary(col['np_data'])
        if length.get(col['layer_id']):
            length[col['layer_id']] = max(length[col['layer_id']], accessor_attribute['length'])
        else:
            length[col['layer_id']] = accessor_attribute['length']
        if not layers[col['layer_id']].get('attributes'):
            layers[col['layer_id']]['attributes'] = {}
        layers[col['layer_id']]['attributes'][col['accessor']] = {'value': accessor_attribute['value'], 'dtype': accessor_attribute['dtype'], 'size': accessor_attribute['size']}
    for (layer_key, _) in layers.items():
        layers[layer_key]['length'] = length[layer_key]
    return layers
data_buffer_serialization = dict(to_json=serialize_columns, from_json=None)