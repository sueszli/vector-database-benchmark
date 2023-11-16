from django.contrib.auth.management import create_permissions
from django.db import migrations, models
from django.db.models.functions import Concat, Length, Substr

def add_permissions(apps, schema_editor):
    if False:
        while True:
            i = 10
    app_config = apps.get_app_config('wagtailcore')
    app_config.models_module = True
    create_permissions(app_config, verbosity=0)
    app_config.models_module = None

def populate_grouppagepermission_permission(apps, schema_editor):
    if False:
        return 10
    ContentType = apps.get_model('contenttypes.ContentType')
    Permission = apps.get_model('auth.Permission')
    GroupPagePermission = apps.get_model('wagtailcore.GroupPagePermission')
    page_type = ContentType.objects.get_by_natural_key('wagtailcore', 'page')
    GroupPagePermission.objects.filter(models.Q(permission__isnull=True) | models.Q(permission_type='edit')).annotate(normalised_permission_type=models.Case(models.When(permission_type='edit', then=models.Value('change')), default=models.F('permission_type'))).update(permission=Permission.objects.filter(content_type=page_type, codename=Concat(models.OuterRef('normalised_permission_type'), models.Value('_page'))).values_list('pk', flat=True)[:1], permission_type=models.F('normalised_permission_type'))

def revert_grouppagepermission_permission(apps, schema_editor):
    if False:
        return 10
    GroupPagePermission = apps.get_model('wagtailcore.GroupPagePermission')
    Permission = apps.get_model('auth.Permission')
    permission_type = Permission.objects.filter(pk=models.OuterRef('permission')).annotate(action=Substr(models.F('codename'), 1, Length(models.F('codename')) - 5)).annotate(permission_type=models.Case(models.When(action='change', then=models.Value('edit')), default=models.F('action'))).values('permission_type')[:1]
    GroupPagePermission.objects.all().update(permission_type=permission_type, permission=None)

class Migration(migrations.Migration):
    dependencies = [('wagtailcore', '0085_add_grouppagepermission_permission')]
    operations = [migrations.RunPython(add_permissions, migrations.operations.RunPython.noop), migrations.RunPython(populate_grouppagepermission_permission, revert_grouppagepermission_permission)]