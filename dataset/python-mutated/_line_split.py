from __future__ import annotations
import re
LINE_AND_ENDING_PATTERN = re.compile('(.*?)(\\r\\n|\\r|\\n|$)', re.S)

def line_split(input_string: str) -> list[tuple[str, str]]:
    if False:
        print('Hello World!')
    '\n    Splits an arbitrary string into a list of tuples, where each tuple contains a line of text and its line ending.\n\n    Args:\n        input_string (str): The string to split.\n\n    Returns:\n        list[tuple[str, str]]: A list of tuples, where each tuple contains a line of text and its line ending.\n\n    Example:\n        split_string_to_lines_and_endings("Hello\\r\\nWorld\\nThis is a test\\rLast line")\n        >>> [(\'Hello\', \'\\r\\n\'), (\'World\', \'\\n\'), (\'This is a test\', \'\\r\'), (\'Last line\', \'\')]\n    '
    return LINE_AND_ENDING_PATTERN.findall(input_string)[:-1] if input_string else []