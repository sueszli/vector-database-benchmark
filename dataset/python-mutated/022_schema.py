SCHEMA_VERSION = 22

def _fix_payer_address(database):
    if False:
        while True:
            i = 10
    database.execute_sql("UPDATE income SET payer_address = '0x' || payer_address WHERE payer_address NOT LIKE '0x%'")

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        while True:
            i = 10
    migrator.python(_fix_payer_address, database)

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        while True:
            i = 10
    pass