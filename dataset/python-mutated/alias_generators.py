"""Alias generators for converting between different capitalization conventions."""
import re
__all__ = ('to_pascal', 'to_camel', 'to_snake')

def to_pascal(snake: str) -> str:
    if False:
        i = 10
        return i + 15
    'Convert a snake_case string to PascalCase.\n\n    Args:\n        snake: The string to convert.\n\n    Returns:\n        The PascalCase string.\n    '
    camel = snake.title()
    return re.sub('([0-9A-Za-z])_(?=[0-9A-Z])', lambda m: m.group(1), camel)

def to_camel(snake: str) -> str:
    if False:
        i = 10
        return i + 15
    'Convert a snake_case string to camelCase.\n\n    Args:\n        snake: The string to convert.\n\n    Returns:\n        The converted camelCase string.\n    '
    camel = to_pascal(snake)
    return re.sub('(^_*[A-Z])', lambda m: m.group(1).lower(), camel)

def to_snake(camel: str) -> str:
    if False:
        print('Hello World!')
    'Convert a PascalCase or camelCase string to snake_case.\n\n    Args:\n        camel: The string to convert.\n\n    Returns:\n        The converted string in snake_case.\n    '
    snake = re.sub('([a-zA-Z])([0-9])', lambda m: f'{m.group(1)}_{m.group(2)}', camel)
    snake = re.sub('([a-z0-9])([A-Z])', lambda m: f'{m.group(1)}_{m.group(2)}', snake)
    return snake.lower()