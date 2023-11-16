from django.contrib.auth.hashers import make_password
from django.db import migrations

def forwards_func(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    User = apps.get_model('auth', 'User')
    old_password_patterns = ('sha1$', 'md5$', 'crypt$')
    for pattern in old_password_patterns:
        users = User.objects.filter(password__startswith=pattern)
        for user in users:
            user.password = make_password(None)
            user.save()

class Migration(migrations.Migration):
    dependencies = [('core', '0004_ad-opt-out'), ('auth', '0008_alter_user_username_max_length')]
    operations = [migrations.RunPython(forwards_func)]