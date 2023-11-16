""" Responsible for building schema code directory hierarchy based on schema name """
import re
CHARACTER_TO_SANITIZE = '[^a-zA-Z0-9_@]'
POTENTIAL_PACKAGE_SEPARATOR = '[@]'

def get_package_hierarchy(schema_name):
    if False:
        print('Hello World!')
    path = 'schema'
    if schema_name.startswith('aws.partner-'):
        path = path + '.aws.partner'
        tail = schema_name[len('aws.partner-'):]
        path = path + '.' + sanitize_name(tail)
        return path.lower()
    if schema_name.startswith('aws.'):
        parts = schema_name.split('.')
        for part in parts:
            path = path + '.'
            path = path + sanitize_name(part)
        return path.lower()
    return f'{path}.{sanitize_name(schema_name)}'.lower()

def sanitize_name(name):
    if False:
        i = 10
        return i + 15
    name = re.sub(CHARACTER_TO_SANITIZE, '_', name)
    return re.sub(POTENTIAL_PACKAGE_SEPARATOR, '.', name)