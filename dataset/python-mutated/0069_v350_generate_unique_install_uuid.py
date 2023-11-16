from __future__ import unicode_literals
from uuid import uuid4
from django.db import migrations
from django.utils.timezone import now

def _gen_install_uuid(apps, schema_editor):
    if False:
        while True:
            i = 10
    Setting = apps.get_model('conf', 'Setting')
    Setting(key='INSTALL_UUID', value=str(uuid4()), created=now(), modified=now()).save()

class Migration(migrations.Migration):
    dependencies = [('main', '0068_v350_index_event_created')]
    operations = [migrations.RunPython(_gen_install_uuid)]