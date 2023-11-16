from django.contrib.postgres.operations import AddIndexConcurrently
from django.db import connection, migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def backfill_is_user_active(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        i = 10
        return i + 15
    Subscription = apps.get_model('zerver', 'Subscription')
    BATCH_SIZE = 1000
    lower_id_bound = 0
    max_id = Subscription.objects.aggregate(models.Max('id'))['id__max']
    if max_id is None:
        return
    while lower_id_bound <= max_id:
        print(f'Processed {lower_id_bound} / {max_id}')
        upper_id_bound = lower_id_bound + BATCH_SIZE
        with connection.cursor() as cursor:
            cursor.execute('\n                UPDATE zerver_subscription\n                SET is_user_active = zerver_userprofile.is_active\n                FROM zerver_userprofile\n                WHERE zerver_subscription.user_profile_id = zerver_userprofile.id\n                AND zerver_subscription.id BETWEEN %(lower_id_bound)s AND %(upper_id_bound)s\n                ', {'lower_id_bound': lower_id_bound, 'upper_id_bound': upper_id_bound})
        lower_id_bound += BATCH_SIZE + 1

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('zerver', '0312_subscription_is_user_active')]
    operations = [migrations.RunPython(backfill_is_user_active, reverse_code=migrations.RunPython.noop), migrations.AlterField(model_name='subscription', name='is_user_active', field=models.BooleanField()), AddIndexConcurrently(model_name='subscription', index=models.Index(condition=models.Q(('active', True), ('is_user_active', True)), fields=['recipient', 'user_profile'], name='zerver_subscription_recipient_id_user_profile_id_idx'))]