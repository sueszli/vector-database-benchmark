from itertools import chain
from django.db import connection, migrations, OperationalError, ProgrammingError

def squash_data(squashed):
    if False:
        for i in range(10):
            print('nop')
    'Returns a tuple of the squashed_keys and the key position to begin\n    processing replace and operation lists'
    cm = current_migration()
    squashed_keys = sorted(squashed.keys())
    if cm is None:
        return (squashed_keys, 0)
    try:
        key_index = squashed_keys.index(cm.name) + 1
    except ValueError:
        key_index = 0
    return (squashed_keys, key_index)

def current_migration(exclude_squashed=True):
    if False:
        for i in range(10):
            print('nop')
    'Get the latest migration non-squashed migration'
    try:
        recorder = migrations.recorder.MigrationRecorder(connection)
        migration_qs = recorder.migration_qs.filter(app='main')
        if exclude_squashed:
            migration_qs = migration_qs.exclude(name__contains='squashed')
        return migration_qs.latest('id')
    except (recorder.Migration.DoesNotExist, OperationalError, ProgrammingError):
        return None

def replaces(squashed, applied=False):
    if False:
        for i in range(10):
            print('nop')
    'Build a list of replacement migrations based on the most recent non-squashed migration\n    and the provided list of SQUASHED migrations. If the most recent non-squashed migration\n    is not present anywhere in the SQUASHED dictionary, assume they have all been applied.\n\n    If applied is True, this will return a list of all the migrations that have already\n    been applied.\n    '
    (squashed_keys, key_index) = squash_data(squashed)
    if applied:
        return [('main', key) for key in squashed_keys[:key_index]]
    return [('main', key) for key in squashed_keys[key_index:]]

def operations(squashed, applied=False):
    if False:
        print('Hello World!')
    'Build a list of migration operations based on the most recent non-squashed migration\n    and the provided list of squashed migrations. If the most recent non-squashed migration\n    is not present anywhere in the `squashed` dictionary, assume they have all been applied.\n\n    If applied is True, this will return a list of all the operations that have\n    already been applied.\n    '
    (squashed_keys, key_index) = squash_data(squashed)
    op_keys = squashed_keys[:key_index] if applied else squashed_keys[key_index:]
    ops = [squashed[op_key] for op_key in op_keys]
    return [op for op in chain.from_iterable(ops)]