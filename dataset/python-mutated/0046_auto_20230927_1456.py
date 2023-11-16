from django.db import migrations

def migrate_user_default_email(apps, *args):
    if False:
        print('Hello World!')
    user_cls = apps.get_model('users', 'User')
    user_cls.objects.filter(email='admin@mycomany.com').update(email='admin@example.com')

class Migration(migrations.Migration):
    dependencies = [('users', '0045_delete_usersession')]
    operations = [migrations.RunPython(migrate_user_default_email)]