import re

def to_camel_case(snake_str: str) -> str:
    if False:
        while True:
            i = 10
    components = snake_str.split('_')
    return components[0] + ''.join((x.capitalize() if x else '_' for x in components[1:]))
TO_KEBAB_CASE_RE = re.compile('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')

def to_kebab_case(name: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    return TO_KEBAB_CASE_RE.sub('-\\1', name).lower()

def capitalize_first(name: str) -> str:
    if False:
        print('Hello World!')
    return name[0].upper() + name[1:]

def to_snake_case(name: str) -> str:
    if False:
        i = 10
        return i + 15
    name = re.sub('(.)([A-Z][a-z]+)', '\\1_\\2', name)
    return re.sub('([a-z0-9])([A-Z])', '\\1_\\2', name).lower()