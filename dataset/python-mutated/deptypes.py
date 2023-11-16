"""Data structures that represent Spack's edge types."""
from typing import Iterable, List, Tuple, Union
DepFlag = int
DepTypes = Union[str, List[str], Tuple[str, ...]]
DepType = str
LINK = 1
RUN = 2
BUILD = 4
TEST = 8
ALL_TYPES: Tuple[DepType, ...] = ('build', 'link', 'run', 'test')
DEFAULT_TYPES: Tuple[DepType, ...] = ('build', 'link')
ALL: DepFlag = BUILD | LINK | RUN | TEST
DEFAULT: DepFlag = BUILD | LINK
ALL_FLAGS: Tuple[DepFlag, DepFlag, DepFlag, DepFlag] = (BUILD, LINK, RUN, TEST)

def flag_from_string(s: str) -> DepFlag:
    if False:
        i = 10
        return i + 15
    if s == 'build':
        return BUILD
    elif s == 'link':
        return LINK
    elif s == 'run':
        return RUN
    elif s == 'test':
        return TEST
    else:
        raise ValueError(f'Invalid dependency type: {s}')

def flag_from_strings(deptype: Iterable[str]) -> DepFlag:
    if False:
        return 10
    'Transform an iterable of deptype strings into a flag.'
    flag = 0
    for deptype_str in deptype:
        flag |= flag_from_string(deptype_str)
    return flag

def canonicalize(deptype: DepTypes) -> DepFlag:
    if False:
        return 10
    "Convert deptype user input to a DepFlag, or raise ValueError.\n\n    Args:\n        deptype: string representing dependency type, or a list/tuple of such strings.\n            Can also be the builtin function ``all`` or the string 'all', which result in\n            a tuple of all dependency types known to Spack.\n    "
    if deptype in ('all', all):
        return ALL
    if isinstance(deptype, str):
        return flag_from_string(deptype)
    if isinstance(deptype, (tuple, list, set)):
        return flag_from_strings(deptype)
    raise ValueError(f'Invalid dependency type: {deptype!r}')

def flag_to_tuple(x: DepFlag) -> Tuple[DepType, ...]:
    if False:
        return 10
    deptype: List[DepType] = []
    if x & BUILD:
        deptype.append('build')
    if x & LINK:
        deptype.append('link')
    if x & RUN:
        deptype.append('run')
    if x & TEST:
        deptype.append('test')
    return tuple(deptype)

def flag_to_string(x: DepFlag) -> DepType:
    if False:
        for i in range(10):
            print('nop')
    if x == BUILD:
        return 'build'
    elif x == LINK:
        return 'link'
    elif x == RUN:
        return 'run'
    elif x == TEST:
        return 'test'
    else:
        raise ValueError(f'Invalid dependency type flag: {x}')

def flag_to_chars(depflag: DepFlag) -> str:
    if False:
        return 10
    "Create a string representing deptypes for many dependencies.\n\n    The string will be some subset of 'blrt', like 'bl ', 'b t', or\n    ' lr ' where each letter in 'blrt' stands for 'build', 'link',\n    'run', and 'test' (the dependency types).\n\n    For a single dependency, this just indicates that the dependency has\n    the indicated deptypes. For a list of dependnecies, this shows\n    whether ANY dpeendency in the list has the deptypes (so the deptypes\n    are merged)."
    return ''.join((t_str[0] if t_flag & depflag else ' ' for (t_str, t_flag) in zip(ALL_TYPES, ALL_FLAGS)))