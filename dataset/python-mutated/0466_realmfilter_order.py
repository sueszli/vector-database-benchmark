from django.db import migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from django.db.models import F

def migrate_set_order_value(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        for i in range(10):
            print('nop')
    RealmFilter = apps.get_model('zerver', 'RealmFilter')
    RealmFilter.objects.all().update(order=F('id'))

class Migration(migrations.Migration):
    dependencies = [('zerver', '0465_backfill_scheduledmessagenotificationemail_trigger')]
    operations = [migrations.AddField(model_name='realmfilter', name='order', field=models.IntegerField(default=0)), migrations.RunPython(migrate_set_order_value, reverse_code=migrations.RunPython.noop, elidable=True)]