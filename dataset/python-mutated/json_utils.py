"""Utilities for dealing with writing json strings.

json_utils wraps json.dump and json.dumps so that they can be used to safely
control the precision of floats when writing to json strings or files.
"""
import json
from json import encoder

def Dump(obj, fid, float_digits=-1, **params):
    if False:
        return 10
    'Wrapper of json.dump that allows specifying the float precision used.\n\n  Args:\n    obj: The object to dump.\n    fid: The file id to write to.\n    float_digits: The number of digits of precision when writing floats out.\n    **params: Additional parameters to pass to json.dumps.\n  '
    original_encoder = encoder.FLOAT_REPR
    if float_digits >= 0:
        encoder.FLOAT_REPR = lambda o: format(o, '.%df' % float_digits)
    try:
        json.dump(obj, fid, **params)
    finally:
        encoder.FLOAT_REPR = original_encoder

def Dumps(obj, float_digits=-1, **params):
    if False:
        return 10
    'Wrapper of json.dumps that allows specifying the float precision used.\n\n  Args:\n    obj: The object to dump.\n    float_digits: The number of digits of precision when writing floats out.\n    **params: Additional parameters to pass to json.dumps.\n\n  Returns:\n    output: JSON string representation of obj.\n  '
    original_encoder = encoder.FLOAT_REPR
    original_c_make_encoder = encoder.c_make_encoder
    if float_digits >= 0:
        encoder.FLOAT_REPR = lambda o: format(o, '.%df' % float_digits)
        encoder.c_make_encoder = None
    try:
        output = json.dumps(obj, **params)
    finally:
        encoder.FLOAT_REPR = original_encoder
        encoder.c_make_encoder = original_c_make_encoder
    return output

def PrettyParams(**params):
    if False:
        return 10
    'Returns parameters for use with Dump and Dumps to output pretty json.\n\n  Example usage:\n    ```json_str = json_utils.Dumps(obj, **json_utils.PrettyParams())```\n    ```json_str = json_utils.Dumps(\n                      obj, **json_utils.PrettyParams(allow_nans=False))```\n\n  Args:\n    **params: Additional params to pass to json.dump or json.dumps.\n\n  Returns:\n    params: Parameters that are compatible with json_utils.Dump and\n      json_utils.Dumps.\n  '
    params['float_digits'] = 4
    params['sort_keys'] = True
    params['indent'] = 2
    params['separators'] = (',', ': ')
    return params