from django.db import migrations
from django.db.models import F

class Migration(migrations.Migration):

    def move_to_vuln_id(apps, schema_editor):
        if False:
            return 10
        finding_model = apps.get_model('dojo', 'Finding')
        test_type_model = apps.get_model('dojo', 'Test_Type')
        (anchore_scan, _) = test_type_model.objects.get_or_create(name='Anchore Engine Scan')
        findings = finding_model.objects.filter(test__test_type=anchore_scan, unique_id_from_tool__isnull=False)
        findings.update(vuln_id_from_tool=F('unique_id_from_tool'))
        findings.update(unique_id_from_tool=None)

    def reverse_move_to_vuln_id(apps, schema_editor):
        if False:
            return 10
        finding_model = apps.get_model('dojo', 'Finding')
        test_type_model = apps.get_model('dojo', 'Test_Type')
        (anchore_scan, _) = test_type_model.objects.get_or_create(name='Anchore Engine Scan')
        findings = finding_model.objects.filter(test__test_type=anchore_scan, vuln_id_from_tool__isnull=False)
        findings.update(unique_id_from_tool=F('vuln_id_from_tool'))
        findings.update(vuln_id_from_tool=None)
    dependencies = [('dojo', '0097_engagement_type')]
    operations = [migrations.RunPython(move_to_vuln_id, reverse_move_to_vuln_id)]