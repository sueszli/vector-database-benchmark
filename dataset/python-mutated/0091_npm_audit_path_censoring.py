from dojo.tools.npm_audit.parser import censor_path_hashes
from dojo.utils import mass_model_updater
from django.db import migrations
from django.utils import timezone
import logging
logger = logging.getLogger(__name__)

def censor_hashes(finding):
    if False:
        i = 10
        return i + 15
    finding.file_path = censor_path_hashes(finding.file_path)
    return finding

def npm_censor_hashes(apps, schema_editor):
    if False:
        while True:
            i = 10
    logger.info('Removing random hashes from npm audit file_paths')
    now = timezone.now()
    Finding = apps.get_model('dojo', 'Finding')
    Test_Type = apps.get_model('dojo', 'Test_Type')
    (npm_audit, _) = Test_Type.objects.get_or_create(name='NPM Audit Scan')
    findings = Finding.objects.filter(test__test_type=npm_audit)
    mass_model_updater(Finding, findings, lambda f: censor_hashes(f), fields=['file_path', 'hash_code'])

class Migration(migrations.Migration):
    dependencies = [('dojo', '0090_index_duplicate_finding')]
    operations = [migrations.RunPython(npm_censor_hashes, migrations.RunPython.noop)]