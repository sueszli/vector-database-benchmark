from django.db import migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from django.db.models import F

def copy_email_field(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        i = 10
        return i + 15
    UserProfile = apps.get_model('zerver', 'UserProfile')
    UserProfile.objects.all().update(delivery_email=F('email'))

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('zerver', '0173_support_seat_based_plans')]
    operations = [migrations.AddField(model_name='userprofile', name='delivery_email', field=models.EmailField(db_index=True, default='', max_length=254), preserve_default=False), migrations.RunPython(copy_email_field, reverse_code=migrations.RunPython.noop, elidable=True)]