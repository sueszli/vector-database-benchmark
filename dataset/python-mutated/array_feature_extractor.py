from six import integer_types as _integer_types
from . import datatypes
from .. import SPECIFICATION_VERSION
from ..proto import Model_pb2 as _Model_pb2
from ..proto import FeatureTypes_pb2 as _FeatureTypes_pb2
from ._interface_management import set_transform_interface_params

def create_array_feature_extractor(input_features, output_name, extract_indices, output_type=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Creates a feature extractor from an input array feature, return\n\n    input_features is a list of one (name, array) tuple.\n\n    extract_indices is either an integer or a list.  If it's an integer,\n    the output type is by default a double (but may also be an integer).\n    If a list, the output type is an array.\n    "
    assert len(input_features) == 1
    assert isinstance(input_features[0][1], datatypes.Array)
    spec = _Model_pb2.Model()
    spec.specificationVersion = SPECIFICATION_VERSION
    if isinstance(extract_indices, _integer_types):
        extract_indices = [extract_indices]
        if output_type is None:
            output_type = datatypes.Double()
    elif isinstance(extract_indices, (list, tuple)):
        if not all((isinstance(x, _integer_types) for x in extract_indices)):
            raise TypeError('extract_indices must be an integer or a list of integers.')
        if output_type is None:
            output_type = datatypes.Array(len(extract_indices))
    else:
        raise TypeError('extract_indices must be an integer or a list of integers.')
    output_features = [(output_name, output_type)]
    for idx in extract_indices:
        assert idx < input_features[0][1].num_elements
        spec.arrayFeatureExtractor.extractIndex.append(idx)
    set_transform_interface_params(spec, input_features, output_features)
    return spec