import logging
from golem.model import UTCDateTimeField, default_now
SCHEMA_VERSION = 49
logger = logging.getLogger('golem.database')

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        while True:
            i = 10
    docker_table = 'dockerwhitelist'
    column_names = [x.name for x in database.get_columns(docker_table)]
    if 'created_date' in column_names:
        logger.info('created_date already exist. skipping migration')
        return
    migrator.add_fields(docker_table, created_date=UTCDateTimeField(default=default_now), modified_date=UTCDateTimeField(default=default_now))

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        while True:
            i = 10
    pass