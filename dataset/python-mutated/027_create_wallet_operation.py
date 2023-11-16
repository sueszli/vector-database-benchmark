import datetime
import peewee as pw
SCHEMA_VERSION = 27

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        i = 10
        return i + 15

    @migrator.create_model
    class WalletOperation(pw.Model):
        tx_hash = pw.CharField()
        direction = pw.CharField()
        status = pw.CharField()
        sender_address = pw.CharField()
        recipient_address = pw.CharField()
        amount = pw.CharField()
        currency = pw.CharField()
        gas_cost = pw.CharField()
        created_date = pw.DateTimeField(default=datetime.datetime.now)
        modified_date = pw.DateTimeField(default=datetime.datetime.now)

        class Meta:
            db_table = 'walletoperation'

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        i = 10
        return i + 15
    migrator.remove_model('walletoperation')