from django.apps import apps as registry
from django.db import migrations
from django.db.models.signals import post_migrate

def assign_permissions(apps, schema_editor):
    if False:
        i = 10
        return i + 15

    def on_migrations_complete(sender=None, **kwargs):
        if False:
            print('Hello World!')
        try:
            apps = kwargs['apps']
        except KeyError:
            return
        Permission = apps.get_model('permission', 'Permission')
        App = apps.get_model('app', 'App')
        Group = apps.get_model('account', 'Group')
        ContentType = apps.get_model('contenttypes', 'ContentType')
        (ct, _) = ContentType.objects.get_or_create(app_label='checkout', model='checkout')
        (handle_checkouts, _) = Permission.objects.get_or_create(name='Handle checkouts', content_type=ct, codename='handle_checkouts')
        manage_checkouts = Permission.objects.filter(codename='manage_checkouts', content_type__app_label='checkout').first()
        app_qs = App.objects.filter(permissions=manage_checkouts)
        for app in app_qs.iterator():
            app.permissions.add(handle_checkouts)
        groups = Group.objects.filter(permissions=manage_checkouts)
        for group in groups.iterator():
            group.permissions.add(handle_checkouts)
    sender = registry.get_app_config('checkout')
    post_migrate.connect(on_migrations_complete, weak=False, sender=sender)

class Migration(migrations.Migration):
    dependencies = [('product', '0159_auto_20220209_1501'), ('order', '0133_rename_order_token_id'), ('checkout', '0039_alter_checkout_email')]
    operations = [migrations.AlterModelOptions(name='checkout', options={'ordering': ('-last_change', 'pk'), 'permissions': (('manage_checkouts', 'Manage checkouts'), ('handle_checkouts', 'Handle checkouts'))}), migrations.RunPython(assign_permissions, migrations.RunPython.noop)]