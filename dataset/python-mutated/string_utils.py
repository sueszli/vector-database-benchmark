"""Common string utilities used by various parts of core."""
import re

def camel_case_split(identifier: str) -> str:
    if False:
        return 10
    'Split camel case string.'
    regex = '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)'
    matches = re.finditer(regex, identifier)
    return ' '.join([m.group(0) for m in matches])