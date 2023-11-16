from django.apps import apps as registry
from django.db import migrations
from django.db.models.signals import post_migrate
from ...core.search_tasks import set_user_search_document_values

def update_user_search_document_values(apps, _schema_editor):
    if False:
        return 10

    def on_migrations_complete(sender=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        set_user_search_document_values.delay()
    sender = registry.get_app_config('account')
    post_migrate.connect(on_migrations_complete, weak=False, sender=sender)

class Migration(migrations.Migration):
    dependencies = [('account', '0057_user_search_document')]
    operations = [migrations.RunPython(update_user_search_document_values, reverse_code=migrations.RunPython.noop)]