import peewee as pw
SCHEMA_VERSION = 30

def _copy_tx_hash(database):
    if False:
        return 10
    database.execute_sql('UPDATE walletoperation SET null_tx_hash=tx_hash')

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    migrator.add_fields('walletoperation', null_tx_hash=pw.CharField(null=True))
    migrator.python(_copy_tx_hash, database)
    migrator.remove_fields('walletoperation', 'tx_hash')
    migrator.rename_field('walletoperation', 'null_tx_hash', 'tx_hash')
    migrator.remove_fields('taskpayment', 'accepted_ts', 'settled_ts')
    migrator.add_fields('taskpayment', accepted_ts=pw.IntegerField(null=True, index=True), settled_ts=pw.IntegerField(null=True))

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        i = 10
        return i + 15
    migrator.add_fields('walletoperation', not_null_tx_hash=pw.CharField(null=True))
    database.execute_sql('UPDATE walletoperation SET not_null_tx_hash=tx_hash')
    migrator.remove_fields('walletoperation', 'tx_hash')
    migrator.rename_field('walletoperation', 'not_null_tx_hash', 'tx_hash')
    migrator.remove_fields('taskpayment', 'accepted_ts', 'settled_ts')
    migrator.add_fields('taskpayment', accepted_ts=pw.DateTimeField(), settled_ts=pw.DateTimeField())