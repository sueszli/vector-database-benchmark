from itertools import chain
from faker.typing import OrderedDictType

def add_ordereddicts(*odicts: OrderedDictType) -> OrderedDictType:
    if False:
        for i in range(10):
            print('nop')
    items = [odict.items() for odict in odicts]
    return OrderedDictType(chain(*items))