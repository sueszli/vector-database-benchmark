from __future__ import unicode_literals
import logging
from django.db import migrations
from awx.main.migrations._inventory_source import set_new_instance_id, back_out_new_instance_id
logger = logging.getLogger('awx.main.migrations')
GCE_INSTANCE_ID_VAR = 'gce_id'

def gce_id_forward(apps, schema_editor):
    if False:
        return 10
    set_new_instance_id(apps, 'gce', GCE_INSTANCE_ID_VAR)

def gce_id_backward(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    back_out_new_instance_id(apps, 'gce', GCE_INSTANCE_ID_VAR)

class Migration(migrations.Migration):
    dependencies = [('main', '0069_v350_generate_unique_install_uuid')]
    operations = [migrations.RunPython(gce_id_forward, gce_id_backward)]