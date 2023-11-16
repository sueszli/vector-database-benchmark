import common.utils.django
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone
import uuid

def migrate_node_permissions(apps, schema_editor):
    if False:
        while True:
            i = 10
    node_perm_model = apps.get_model('perms', 'NodePermission')
    asset_perm_model = apps.get_model('perms', 'AssetPermission')
    db_alias = schema_editor.connection.alias
    for old in node_perm_model.objects.using(db_alias).all():
        perm = asset_perm_model.objects.using(db_alias).create(name='{}-{}-{}'.format(old.node.value, old.user_group.name, old.system_user.name), is_active=old.is_active, date_expired=old.date_expired, created_by=old.date_expired, date_created=old.date_created, comment=old.comment)
        perm.user_groups.add(old.user_group)
        perm.nodes.add(old.node)
        perm.system_users.add(old.system_user)

def migrate_system_assets_relation(apps, schema_editor):
    if False:
        print('Hello World!')
    system_user_model = apps.get_model('assets', 'SystemUser')
    db_alias = schema_editor.connection.alias
    for s in system_user_model.objects.using(db_alias).all():
        nodes = list(s.nodes.all())
        s.nodes.set([])
        s.nodes.set(nodes)

class Migration(migrations.Migration):
    replaces = [('perms', '0002_auto_20171228_0025'), ('perms', '0003_auto_20180225_1815'), ('perms', '0004_auto_20180411_1135'), ('perms', '0005_migrate_data_20180411_1144'), ('perms', '0006_auto_20180606_1505'), ('perms', '0007_auto_20180807_1116'), ('perms', '0008_auto_20180816_1652'), ('perms', '0009_auto_20180903_1132')]
    dependencies = [('users', '0002_auto_20171225_1157'), ('assets', '0007_auto_20180225_1815'), ('assets', '0013_auto_20180411_1135'), ('users', '0004_auto_20180125_1218'), ('perms', '0001_initial'), migrations.swappable_dependency(settings.AUTH_USER_MODEL)]
    operations = [migrations.AddField(model_name='assetpermission', name='user_groups', field=models.ManyToManyField(blank=True, related_name='asset_permissions', to='users.UserGroup', verbose_name='User group')), migrations.AddField(model_name='assetpermission', name='users', field=models.ManyToManyField(blank=True, related_name='asset_permissions', to=settings.AUTH_USER_MODEL, verbose_name='User')), migrations.RemoveField(model_name='assetpermission', name='asset_groups'), migrations.AddField(model_name='assetpermission', name='date_start', field=models.DateTimeField(default=django.utils.timezone.now, verbose_name='Date start')), migrations.AddField(model_name='assetpermission', name='nodes', field=models.ManyToManyField(blank=True, related_name='granted_by_permissions', to='assets.Node', verbose_name='Nodes')), migrations.RunPython(code=migrate_system_assets_relation), migrations.AlterField(model_name='assetpermission', name='date_expired', field=models.DateTimeField(db_index=True, default=common.utils.django.date_expired_default, verbose_name='Date expired')), migrations.AlterField(model_name='assetpermission', name='date_start', field=models.DateTimeField(db_index=True, default=django.utils.timezone.now, verbose_name='Date start')), migrations.AddField(model_name='assetpermission', name='org_id', field=models.CharField(blank=True, default=None, max_length=36, null=True)), migrations.AlterField(model_name='assetpermission', name='name', field=models.CharField(max_length=128, verbose_name='Name')), migrations.AlterUniqueTogether(name='assetpermission', unique_together={('org_id', 'name')}), migrations.AlterField(model_name='assetpermission', name='org_id', field=models.CharField(blank=True, db_index=True, default='', max_length=36, verbose_name='Organization')), migrations.AlterModelOptions(name='assetpermission', options={'verbose_name': 'Asset permission'})]