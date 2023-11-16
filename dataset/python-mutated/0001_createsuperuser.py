import os
from django.contrib.auth.models import User
from django.db import migrations
from django.db.backends.postgresql.schema import DatabaseSchemaEditor
from django.db.migrations.state import StateApps
import google.auth
from google.cloud import secretmanager

def createsuperuser(apps: StateApps, schema_editor: DatabaseSchemaEditor) -> None:
    if False:
        print('Hello World!')
    '\n    Dynamically create an admin user as part of a migration\n    Password is pulled from Secret Manger (previously created as part of tutorial)\n    '
    if os.getenv('TRAMPOLINE_CI', None):
        admin_password = 'test'
    else:
        client = secretmanager.SecretManagerServiceClient()
        (_, project) = google.auth.default()
        PASSWORD_NAME = os.environ.get('PASSWORD_NAME', 'superuser_password')
        name = f'projects/{project}/secrets/{PASSWORD_NAME}/versions/latest'
        admin_password = client.access_secret_version(name=name).payload.data.decode('UTF-8')
    User.objects.create_superuser('admin', password=admin_password.strip())

class Migration(migrations.Migration):
    initial = True
    dependencies = []
    operations = [migrations.RunPython(createsuperuser)]