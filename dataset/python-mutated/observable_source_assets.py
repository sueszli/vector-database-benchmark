def read_some_file():
    if False:
        i = 10
        return i + 15
    return 'foo'
from hashlib import sha256
from dagster import DataVersion, observable_source_asset

@observable_source_asset
def foo_source_asset():
    if False:
        for i in range(10):
            print('nop')
    content = read_some_file()
    hash_sig = sha256()
    hash_sig.update(bytearray(content, 'utf8'))
    return DataVersion(hash_sig.hexdigest())
from dagster import DataVersion, ScheduleDefinition, define_asset_job, observable_source_asset

@observable_source_asset
def foo_source_asset():
    if False:
        while True:
            i = 10
    content = read_some_file()
    hash_sig = sha256()
    hash_sig.update(bytearray(content, 'utf8'))
    return DataVersion(hash_sig.hexdigest())
observation_job = define_asset_job('observation_job', [foo_source_asset])
observation_schedule = ScheduleDefinition(name='observation_schedule', cron_schedule='@daily', job=observation_job)