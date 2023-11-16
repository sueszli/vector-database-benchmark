import warnings
from collections import Counter
from itertools import chain
from typing import Tuple
import strawberry
from strawberry.type import has_object_definition

def merge_types(name: str, types: Tuple[type, ...]) -> type:
    if False:
        return 10
    'Merge multiple Strawberry types into one\n\n    For example, given two queries `A` and `B`, one can merge them into a\n    super type as follows:\n\n        merge_types("SuperQuery", (B, A))\n\n    This is essentially the same as:\n\n        class SuperQuery(B, A):\n            ...\n    '
    if not types:
        raise ValueError("Can't merge types if none are supplied")
    fields = chain(*(t.__strawberry_definition__.fields for t in types if has_object_definition(t)))
    counter = Counter((f.name for f in fields))
    dupes = [f for (f, c) in counter.most_common() if c > 1]
    if dupes:
        warnings.warn('{} has overridden fields: {}'.format(name, ', '.join(dupes)), stacklevel=2)
    return strawberry.type(type(name, types, {}))