# Generated by Django 2.2.4 on 2019-09-16 23:50

from django.db import migrations, models
import django.db.models.deletion

from awx.main.models import CredentialType
from awx.main.utils.common import set_current_apps


def create_new_credential_types(apps, schema_editor):
    set_current_apps(apps)
    CredentialType.setup_tower_managed_defaults(apps)


class Migration(migrations.Migration):
    dependencies = [
        ('main', '0095_v360_increase_instance_version_length'),
    ]

    operations = [
        migrations.AddField(
            model_name='instancegroup',
            name='credential',
            field=models.ForeignKey(
                blank=True, default=None, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='instancegroups', to='main.Credential'
            ),
        ),
        migrations.AddField(
            model_name='instancegroup',
            name='pod_spec_override',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AlterField(
            model_name='credentialtype',
            name='kind',
            field=models.CharField(
                choices=[
                    ('ssh', 'Machine'),
                    ('vault', 'Vault'),
                    ('net', 'Network'),
                    ('scm', 'Source Control'),
                    ('cloud', 'Cloud'),
                    ('token', 'Personal Access Token'),
                    ('insights', 'Insights'),
                    ('external', 'External'),
                    ('kubernetes', 'Kubernetes'),
                ],
                max_length=32,
            ),
        ),
        migrations.RunPython(create_new_credential_types),
    ]
