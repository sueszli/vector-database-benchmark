from pandas import DataFrame
from typing import Dict, Tuple
import re
"\nResolves any dependencies and requirements before\na cleaning action is made. All dependency resolution functions return both\n- a boolean value describing whether all dependencies are resolved\n- a string message describing the error in the case that dependencies aren't resolved\n"

def default_resolution(df: DataFrame, action: Dict) -> Tuple[bool, str]:
    if False:
        return 10
    return (True, None)

def resolve_filter_action(df: DataFrame, action: Dict) -> Tuple[bool, str]:
    if False:
        for i in range(10):
            print('nop')
    for name in df.columns:
        if re.search('\\s', name):
            return (False, 'Column name contains whitespace or newline characters which cannot be used in filter actions')
    return (True, None)