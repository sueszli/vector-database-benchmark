from __future__ import unicode_literals
from django.db import migrations
from awx.conf.migrations import _rename_setting

def copy_allowed_ips(apps, schema_editor):
    if False:
        while True:
            i = 10
    _rename_setting.rename_setting(apps, schema_editor, old_key='PROXY_IP_WHITELIST', new_key='PROXY_IP_ALLOWED_LIST')

class Migration(migrations.Migration):
    dependencies = [('conf', '0006_v331_ldap_group_type')]
    operations = [migrations.RunPython(copy_allowed_ips)]