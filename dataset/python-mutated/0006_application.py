from django.db import migrations, models
import django.db.models.deletion
import uuid
CATEGORY_DB_LIST = ['mysql', 'oracle', 'postgresql', 'mariadb']
CATEGORY_REMOTE_LIST = ['chrome', 'mysql_workbench', 'vmware_client', 'custom']
CATEGORY_CLOUD_LIST = ['k8s']
CATEGORY_DB = 'db'
CATEGORY_REMOTE = 'remote_app'
CATEGORY_CLOUD = 'cloud'
CATEGORY_LIST = [CATEGORY_DB, CATEGORY_REMOTE, CATEGORY_CLOUD]

def get_application_category(old_app):
    if False:
        return 10
    _type = old_app.type
    if _type in CATEGORY_DB_LIST:
        category = CATEGORY_DB
    elif _type in CATEGORY_REMOTE_LIST:
        category = CATEGORY_REMOTE
    elif _type in CATEGORY_CLOUD_LIST:
        category = CATEGORY_CLOUD
    else:
        category = None
    return category

def common_to_application_json(old_app):
    if False:
        for i in range(10):
            print('nop')
    category = get_application_category(old_app)
    date_updated = old_app.date_updated if hasattr(old_app, 'date_updated') else old_app.date_created
    return {'id': old_app.id, 'name': old_app.name, 'type': old_app.type, 'category': category, 'comment': old_app.comment, 'created_by': old_app.created_by, 'date_created': old_app.date_created, 'date_updated': date_updated, 'org_id': old_app.org_id}

def db_to_application_json(database):
    if False:
        for i in range(10):
            print('nop')
    app_json = common_to_application_json(database)
    app_json.update({'attrs': {'host': database.host, 'port': database.port, 'database': database.database}})
    return app_json

def remote_to_application_json(remote):
    if False:
        for i in range(10):
            print('nop')
    app_json = common_to_application_json(remote)
    attrs = {'asset': str(remote.asset.id), 'path': remote.path}
    attrs.update(remote.params)
    app_json.update({'attrs': attrs})
    return app_json

def k8s_to_application_json(k8s):
    if False:
        i = 10
        return i + 15
    app_json = common_to_application_json(k8s)
    app_json.update({'attrs': {'cluster': k8s.cluster}})
    return app_json

def migrate_and_integrate_applications(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    db_alias = schema_editor.connection.alias
    database_app_model = apps.get_model('applications', 'DatabaseApp')
    remote_app_model = apps.get_model('applications', 'RemoteApp')
    k8s_app_model = apps.get_model('applications', 'K8sApp')
    database_apps = database_app_model.objects.using(db_alias).all()
    remote_apps = remote_app_model.objects.using(db_alias).all()
    k8s_apps = k8s_app_model.objects.using(db_alias).all()
    database_applications = [db_to_application_json(db_app) for db_app in database_apps]
    remote_applications = [remote_to_application_json(remote_app) for remote_app in remote_apps]
    k8s_applications = [k8s_to_application_json(k8s_app) for k8s_app in k8s_apps]
    applications_json = database_applications + remote_applications + k8s_applications
    application_model = apps.get_model('applications', 'Application')
    applications = [application_model(**application_json) for application_json in applications_json if application_json['category'] in CATEGORY_LIST]
    for application in applications:
        if application_model.objects.using(db_alias).filter(name=application.name).exists():
            application.name = '{}-{}'.format(application.name, application.type)
        application.save()

class Migration(migrations.Migration):
    dependencies = [('assets', '0057_fill_node_value_assets_amount_and_parent_key'), ('applications', '0005_k8sapp')]
    operations = [migrations.CreateModel(name='Application', fields=[('org_id', models.CharField(blank=True, db_index=True, default='', max_length=36, verbose_name='Organization')), ('id', models.UUIDField(default=uuid.uuid4, primary_key=True, serialize=False)), ('created_by', models.CharField(blank=True, max_length=32, null=True, verbose_name='Created by')), ('date_created', models.DateTimeField(auto_now_add=True, null=True, verbose_name='Date created')), ('date_updated', models.DateTimeField(auto_now=True, verbose_name='Date updated')), ('name', models.CharField(max_length=128, verbose_name='Name')), ('category', models.CharField(choices=[('db', 'Database'), ('remote_app', 'Remote app'), ('cloud', 'Cloud')], max_length=16, verbose_name='Category')), ('type', models.CharField(choices=[('mysql', 'MySQL'), ('oracle', 'Oracle'), ('postgresql', 'PostgreSQL'), ('mariadb', 'MariaDB'), ('chrome', 'Chrome'), ('mysql_workbench', 'MySQL Workbench'), ('vmware_client', 'vSphere Client'), ('custom', 'Custom'), ('k8s', 'Kubernetes')], max_length=16, verbose_name='Type')), ('attrs', models.JSONField(default=dict)), ('comment', models.TextField(blank=True, default='', max_length=128, verbose_name='Comment')), ('domain', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='applications', to='assets.Domain', verbose_name='Domain'))], options={'ordering': ('name',), 'unique_together': {('org_id', 'name')}}), migrations.RunPython(migrate_and_integrate_applications)]