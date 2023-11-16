import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models
from common.db.utils import Encryptor

def migrate_secret_key(apps, *args):
    if False:
        print('Hello World!')
    user_model = apps.get_model('users', 'User')
    preference_model = apps.get_model('users', 'Preference')
    data = user_model.objects.filter(secret_key__isnull=False).values_list('id', 'secret_key')
    objs = []
    for (user_id, secret_key) in data:
        secret_key = Encryptor(secret_key).encrypt()
        objs.append(preference_model(name='secret_key', category='lina', value=secret_key, encrypted=True, user_id=user_id))
    preference_model.objects.bulk_create(objs)

def migrate_graphical_resolution(apps, *args):
    if False:
        print('Hello World!')
    user_model = apps.get_model('users', 'User')
    setting_model = apps.get_model('settings', 'Setting')
    preference_model = apps.get_model('users', 'Preference')
    s = setting_model.objects.filter(name='TERMINAL_GRAPHICAL_RESOLUTION').first()
    if s and s.value == 'Auto' or not s:
        return
    value = s.value
    objs = []
    for _id in user_model.objects.values_list('id', flat=True):
        objs.append(preference_model(name='rdp_resolution', category='luna', value=value, user_id=_id))
    preference_model.objects.bulk_create(objs)

class Migration(migrations.Migration):
    dependencies = [('users', '0042_auto_20230203_1201')]
    operations = [migrations.CreateModel(name='Preference', fields=[('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')), ('name', models.CharField(max_length=128, verbose_name='Name')), ('category', models.CharField(max_length=128, verbose_name='Category')), ('value', models.TextField(blank=True, null=True, verbose_name='Value')), ('encrypted', models.BooleanField(default=False, verbose_name='Encrypted')), ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='preferences', to=settings.AUTH_USER_MODEL, verbose_name='Users'))], options={'verbose_name': 'Preference', 'db_table': 'users_preference', 'unique_together': {('name', 'user_id')}}), migrations.RunPython(migrate_secret_key), migrations.RunPython(migrate_graphical_resolution), migrations.RemoveField(model_name='user', name='secret_key')]