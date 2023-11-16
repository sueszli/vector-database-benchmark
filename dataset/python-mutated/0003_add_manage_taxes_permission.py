from django.apps import apps as registry
from django.db import migrations
from django.db.models.signals import post_migrate

def assign_permissions(apps, schema_editor):
    if False:
        while True:
            i = 10

    def on_migrations_complete(sender=None, **kwargs):
        if False:
            print('Hello World!')
        try:
            apps = kwargs['apps']
        except KeyError:
            return
        Permission = apps.get_model('permission', 'Permission')
        Group = apps.get_model('account', 'Group')
        ContentType = apps.get_model('contenttypes', 'ContentType')
        (ct, _) = ContentType.objects.get_or_create(app_label='checkout', model='checkout')
        (manage_taxes, _) = Permission.objects.get_or_create(name='Manage taxes', content_type=ct, codename='manage_taxes')
        groups = Group.objects.filter(permissions__content_type__app_label='site', permissions__codename='manage_settings')
        for group in groups:
            group.permissions.add(manage_taxes)
    sender = registry.get_app_config('tax')
    post_migrate.connect(on_migrations_complete, weak=False, sender=sender)

class Migration(migrations.Migration):
    dependencies = [('tax', '0002_add_default_tax_configs')]
    operations = [migrations.RunPython(assign_permissions, migrations.RunPython.noop)]