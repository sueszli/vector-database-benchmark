"""An implementation of RFC4013 SASLprep."""
__all__ = ['saslprep']
import stringprep
from typing import Callable, Tuple
import unicodedata
_PROHIBITED: Tuple[Callable[[str], bool], ...] = (stringprep.in_table_c12, stringprep.in_table_c21_c22, stringprep.in_table_c3, stringprep.in_table_c4, stringprep.in_table_c5, stringprep.in_table_c6, stringprep.in_table_c7, stringprep.in_table_c8, stringprep.in_table_c9)

def saslprep(data: str, prohibit_unassigned_code_points: bool=True) -> str:
    if False:
        for i in range(10):
            print('nop')
    "An implementation of RFC4013 SASLprep.\n    :param data:\n        The string to SASLprep.\n    :param prohibit_unassigned_code_points:\n        RFC 3454 and RFCs for various SASL mechanisms distinguish between\n        `queries` (unassigned code points allowed) and\n        `stored strings` (unassigned code points prohibited). Defaults\n        to ``True`` (unassigned code points are prohibited).\n    :return: The SASLprep'ed version of `data`.\n    "
    if prohibit_unassigned_code_points:
        prohibited = _PROHIBITED + (stringprep.in_table_a1,)
    else:
        prohibited = _PROHIBITED
    in_table_c12 = stringprep.in_table_c12
    in_table_b1 = stringprep.in_table_b1
    data = ''.join([' ' if in_table_c12(elt) else elt for elt in data if not in_table_b1(elt)])
    data = unicodedata.ucd_3_2_0.normalize('NFKC', data)
    in_table_d1 = stringprep.in_table_d1
    if in_table_d1(data[0]):
        if not in_table_d1(data[-1]):
            raise ValueError('SASLprep: failed bidirectional check')
        prohibited = prohibited + (stringprep.in_table_d2,)
    else:
        prohibited = prohibited + (in_table_d1,)
    for char in data:
        if any((in_table(char) for in_table in prohibited)):
            raise ValueError('SASLprep: failed prohibited character check')
    return data