from django.apps import apps as registry
from django.db import migrations
from django.db.models.signals import post_migrate
from ...core.search_tasks import set_order_search_document_values

def update_order_search_document_values(apps, _schema_editor):
    if False:
        for i in range(10):
            print('nop')

    def on_migrations_complete(sender=None, **kwargs):
        if False:
            return 10
        set_order_search_document_values.delay()
    sender = registry.get_app_config('order')
    post_migrate.connect(on_migrations_complete, weak=False, sender=sender)

class Migration(migrations.Migration):
    dependencies = [('order', '0122_merge_20211220_1641'), ('payment', '0030_auto_20210908_1346')]
    operations = [migrations.RunPython(update_order_search_document_values, reverse_code=migrations.RunPython.noop)]