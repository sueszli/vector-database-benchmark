import peewee as pw
from golem.model import Performance
SCHEMA_VERSION = 41

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        while True:
            i = 10
    database.truncate_table(Performance)
    migrator.change_columns(Performance, cpu_usage=pw.IntegerField(default=Performance.DEFAULT_CPU_USAGE))

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    database.truncate_table(Performance)
    migrator.change_columns(Performance, cpu_usage=pw.IntegerField(default=0))