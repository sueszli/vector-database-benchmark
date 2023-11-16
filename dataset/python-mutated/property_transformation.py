from collections import defaultdict
from typing import Iterable, Iterator, NamedTuple

class TransformationResult(NamedTuple):
    source_name: str
    transformed_name: str

def transform_property_names(property_names: Iterable[str]) -> Iterator[TransformationResult]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Transform property names using this rules:\n    1. Remove leading "$" from property_name\n    2. Resolve naming conflicts, like `userName` and `username`,\n    that will break normalization in the future, by adding `_userName`to property name\n    '
    lowercase_collision_count = defaultdict(int)
    lowercase_properties = set()
    for property_name in sorted(property_names):
        property_name_transformed = property_name
        if property_name_transformed.startswith('$'):
            property_name_transformed = property_name_transformed[1:]
        lowercase_property_name = property_name_transformed.lower()
        if lowercase_property_name in lowercase_properties:
            lowercase_collision_count[lowercase_property_name] += 1
            prefix = '_' * lowercase_collision_count[lowercase_property_name]
            property_name_transformed = prefix + property_name_transformed
        lowercase_properties.add(lowercase_property_name)
        yield TransformationResult(source_name=property_name, transformed_name=property_name_transformed)