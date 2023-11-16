from django.db import migrations
import logging
logger = logging.getLogger(__name__)
NESSUS_REFERENCES = ['Nessus Scan', 'Nessus WAS Scan']

def update_test(test, tenable_test_type) -> None:
    if False:
        return 10
    if test.test_type.name in NESSUS_REFERENCES or test.scan_type in NESSUS_REFERENCES:
        test.test_type = tenable_test_type
        test.scan_type = tenable_test_type.name
        test.save()

def update_finding(finding, tenable_test_type, nessus_test_type, nessus_was_test_type) -> None:
    if False:
        print('Hello World!')
    if nessus_test_type in finding.found_by.all():
        finding.found_by.remove(nessus_test_type.id)
    if nessus_was_test_type in finding.found_by.all():
        finding.found_by.remove(nessus_was_test_type.id)
    if tenable_test_type not in finding.found_by.all():
        finding.found_by.add(tenable_test_type.id)
    finding.save()

def migrate_nessus_findings_to_tenable(apps, schema_editor):
    if False:
        print('Hello World!')
    finding_model = apps.get_model('dojo', 'Finding')
    test_type_model = apps.get_model('dojo', 'Test_Type')
    (tenable_test_type, _) = test_type_model.objects.get_or_create(name='Tenable Scan', active=True)
    nessus_test_type = test_type_model.objects.filter(name='Nessus Scan').first()
    nessus_was_test_type = test_type_model.objects.filter(name='Nessus WAS Scan').first()
    findings = finding_model.objects.filter(test__scan_type__in=NESSUS_REFERENCES)
    logger.warning(f'We identified {findings.count()} Nessus/NessusWAS findings to migrate to Tenable findings')
    for finding in findings:
        update_finding(finding, tenable_test_type, nessus_test_type, nessus_was_test_type)
        update_test(finding.test, tenable_test_type)

class Migration(migrations.Migration):
    dependencies = [('dojo', '0186_system_settings_non_common_password_required')]
    operations = [migrations.RunPython(migrate_nessus_findings_to_tenable)]