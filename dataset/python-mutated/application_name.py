import os
import sys

def get_service_name(argv):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return best-effort guess as to the name of this service\n    '
    for arg in argv:
        if arg == '-m':
            continue
        if 'python' in arg:
            continue
        if 'manage' in arg:
            continue
        if arg.startswith('run_'):
            return arg[len('run_'):]
        return arg

def get_application_name(CLUSTER_HOST_ID, function=''):
    if False:
        for i in range(10):
            print('nop')
    if function:
        function = f'_{function}'
    return f'awx-{os.getpid()}-{get_service_name(sys.argv)}{function}-{CLUSTER_HOST_ID}'[:63]

def set_application_name(DATABASES, CLUSTER_HOST_ID, function=''):
    if False:
        i = 10
        return i + 15
    if not DATABASES or 'default' not in DATABASES:
        return
    if 'sqlite3' in DATABASES['default']['ENGINE']:
        return
    options_dict = DATABASES['default'].setdefault('OPTIONS', dict())
    options_dict['application_name'] = get_application_name(CLUSTER_HOST_ID, function)