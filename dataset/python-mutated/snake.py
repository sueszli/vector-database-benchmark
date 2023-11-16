"""Contains functions dealing with snake case conversions."""
from __future__ import annotations
import re
from typing import Any
_re_camel_to_snake = re.compile('([a-z0-9](?=[A-Z])|[A-Z](?=[A-Z][a-z]))')

def camel_to_snake(name: str) -> str:
    if False:
        while True:
            i = 10
    'Convert ``name`` from camelCase to snake_case.'
    return _re_camel_to_snake.sub('\\1_', name).lower()

def snake_case_keys(dictionary: dict[str, Any]) -> dict[str, Any]:
    if False:
        return 10
    'Return a new dictionary with keys converted to snake_case.\n\n    :param dictionary: The dict to be corrected.\n\n    '
    return {camel_to_snake(k): v for (k, v) in dictionary.items()}