import array
from collections import OrderedDict
import numpy as np
from fastavro import parse_schema
from apache_beam.typehints import trivial_inference
from apache_beam.typehints import typehints
try:
    import pyarrow as pa
except ImportError:
    pa = None

def infer_element_type(elements):
    if False:
        for i in range(10):
            print('nop')
    'For internal use only; no backwards-compatibility guarantees.\n\n  Infer a Beam type for a list of elements.\n\n  Args:\n    elements (List[Any]): A list of elements for which the type should be\n        inferred.\n\n  Returns:\n    A Beam type encompassing all elements.\n  '
    element_type = typehints.Union[[trivial_inference.instance_to_type(e) for e in elements]]
    return element_type

def infer_typehints_schema(data):
    if False:
        print('Hello World!')
    'For internal use only; no backwards-compatibility guarantees.\n\n  Infer Beam types for tabular data.\n\n  Args:\n    data (List[dict]): A list of dictionaries representing rows in a table.\n\n  Returns:\n    An OrderedDict mapping column names to Beam types.\n  '
    column_data = OrderedDict()
    for row in data:
        for (key, value) in row.items():
            column_data.setdefault(key, []).append(value)
    column_types = OrderedDict([(key, infer_element_type(values)) for (key, values) in column_data.items()])
    return column_types

def infer_avro_schema(data):
    if False:
        return 10
    'For internal use only; no backwards-compatibility guarantees.\n\n  Infer avro schema for tabular data.\n\n  Args:\n    data (List[dict]): A list of dictionaries representing rows in a table.\n\n  Returns:\n    An avro schema object.\n  '
    _typehint_to_avro_type = {type(None): 'null', int: 'int', float: 'double', str: 'string', bytes: 'bytes', np.ndarray: 'bytes', array.array: 'bytes'}

    def typehint_to_avro_type(value):
        if False:
            return 10
        if isinstance(value, typehints.UnionConstraint):
            return sorted((typehint_to_avro_type(union_type) for union_type in value.union_types))
        else:
            return _typehint_to_avro_type[value]
    column_types = infer_typehints_schema(data)
    avro_fields = [{'name': str(key), 'type': typehint_to_avro_type(value)} for (key, value) in column_types.items()]
    schema_dict = {'namespace': 'example.avro', 'name': 'User', 'type': 'record', 'fields': avro_fields}
    return parse_schema(schema_dict)

def infer_pyarrow_schema(data):
    if False:
        i = 10
        return i + 15
    'For internal use only; no backwards-compatibility guarantees.\n\n  Infer PyArrow schema for tabular data.\n\n  Args:\n    data (List[dict]): A list of dictionaries representing rows in a table.\n\n  Returns:\n    A PyArrow schema object.\n  '
    column_data = OrderedDict()
    for row in data:
        for (key, value) in row.items():
            column_data.setdefault(key, []).append(value)
    column_types = OrderedDict([(key, pa.array(value).type) for (key, value) in column_data.items()])
    return pa.schema(list(column_types.items()))