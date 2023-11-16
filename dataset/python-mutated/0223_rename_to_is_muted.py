from django.db import migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from django.db.models import Case, Value, When

def set_initial_value_for_is_muted(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        while True:
            i = 10
    Subscription = apps.get_model('zerver', 'Subscription')
    Subscription.objects.update(is_muted=Case(When(in_home_view=True, then=Value(False)), When(in_home_view=False, then=Value(True))))

def reverse_code(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        i = 10
        return i + 15
    Subscription = apps.get_model('zerver', 'Subscription')
    Subscription.objects.update(in_home_view=Case(When(is_muted=True, then=Value(False)), When(is_muted=False, then=Value(True))))

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('zerver', '0222_userprofile_fluid_layout_width')]
    operations = [migrations.AddField(model_name='subscription', name='is_muted', field=models.BooleanField(null=True, default=False)), migrations.RunPython(set_initial_value_for_is_muted, reverse_code=reverse_code, elidable=True), migrations.RemoveField(model_name='subscription', name='in_home_view')]