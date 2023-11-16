import re

def camel_to_snake(s: str) -> str:
    if False:
        while True:
            i = 10
    s = re.sub('(.)([A-Z][a-z]+)', '\\1_\\2', s)
    return re.sub('([a-z0-9])([A-Z])', '\\1_\\2', s).lower()