from featuretools import IdentityFeature
from featuretools.primitives.utils import PrimitivesDeserializer

def test_relationship_path(es):
    if False:
        i = 10
        return i + 15
    value = IdentityFeature(es['log'].ww['value'])
    assert len(value.relationship_path) == 0

def test_serialization(es):
    if False:
        for i in range(10):
            print('nop')
    value = IdentityFeature(es['log'].ww['value'])
    dictionary = {'name': 'value', 'column_name': 'value', 'dataframe_name': 'log'}
    assert dictionary == value.get_arguments()
    assert value == IdentityFeature.from_dictionary(dictionary, es, {}, PrimitivesDeserializer)