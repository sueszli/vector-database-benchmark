"""Test multiple from-imports."""
from typing import List, Text

def f() -> List[Text]:
    if False:
        i = 10
        return i + 15
    return []