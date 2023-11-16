from django.contrib.auth.hashers import make_password
from django.db import migrations

def fill_empty_passwords_on_users(apps, _schema_editor):
    if False:
        i = 10
        return i + 15
    User = apps.get_model('account', 'User')
    User.objects.filter(password='').update(password=make_password(None))

class Migration(migrations.Migration):
    dependencies = [('account', '0075_add_address_metadata')]
    operations = [migrations.RunPython(fill_empty_passwords_on_users, reverse_code=migrations.RunPython.noop)]