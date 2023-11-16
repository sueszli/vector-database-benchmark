from django.db import migrations, transaction
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from django.db.models import Max, Min, OuterRef

def set_default_value_for_can_mention_group(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        i = 10
        return i + 15
    UserGroup = apps.get_model('zerver', 'UserGroup')
    BATCH_SIZE = 1000
    max_id = UserGroup.objects.filter(can_mention_group=None).aggregate(Max('id'))['id__max']
    if max_id is None:
        return
    lower_bound = UserGroup.objects.filter(can_mention_group=None).aggregate(Min('id'))['id__min']
    while lower_bound <= max_id:
        upper_bound = lower_bound + BATCH_SIZE - 1
        print(f'Processing batch {lower_bound} to {upper_bound} for UserGroup')
        with transaction.atomic():
            UserGroup.objects.filter(id__range=(lower_bound, upper_bound), can_mention_group=None, is_system_group=True).update(can_mention_group=UserGroup.objects.filter(name='@role:nobody', realm=OuterRef('realm'), is_system_group=True).values('pk'))
            UserGroup.objects.filter(id__range=(lower_bound, upper_bound), can_mention_group=None, is_system_group=False).update(can_mention_group=UserGroup.objects.filter(name='@role:everyone', realm=OuterRef('realm'), is_system_group=True).values('pk'))
        lower_bound += BATCH_SIZE

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('zerver', '0454_usergroup_can_mention_group')]
    operations = [migrations.RunPython(set_default_value_for_can_mention_group, elidable=True, reverse_code=migrations.RunPython.noop)]