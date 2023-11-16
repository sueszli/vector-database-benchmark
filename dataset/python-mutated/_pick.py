from typing import Optional

def pick_bool(*values: Optional[bool]) -> bool:
    if False:
        i = 10
        return i + 15
    'Pick the first non-none bool or return the last value.\n\n    Args:\n        *values (bool): Any number of boolean or None values.\n\n    Returns:\n        bool: First non-none boolean.\n    '
    assert values, '1 or more values required'
    for value in values:
        if value is not None:
            return value
    return bool(value)