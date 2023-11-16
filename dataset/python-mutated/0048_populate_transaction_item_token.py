from django.contrib.postgres.functions import RandomUUID
from django.db import migrations
from django.db.models import Case, When

def populate_transaction_item_token(apps, schema_editor):
    if False:
        return 10
    TransactionItem = apps.get_model('payment', 'TransactionItem')
    TransactionItem.objects.filter(token__isnull=True).update(token=Case(When(token__isnull=True, then=RandomUUID()), default='token'))

class Migration(migrations.Migration):
    dependencies = [('payment', '0047_merge_20230321_1456')]
    operations = [migrations.RunPython(populate_transaction_item_token, migrations.RunPython.noop)]