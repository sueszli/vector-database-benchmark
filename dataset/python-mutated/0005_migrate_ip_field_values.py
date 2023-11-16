from django.db import migrations

def forwards_func(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    "\n    Post migration ip field from GenericIPAddressField to CharField.\n\n    GenericIPAddressField saves the IP with ``{ip}/{range}`` format.\n    We don't need to show the range to users.\n    "
    AuditLog = apps.get_model('audit', 'AuditLog')
    for auditlog in AuditLog.objects.all().iterator():
        ip = auditlog.ip
        if ip:
            ip = ip.split('/', maxsplit=1)[0]
            auditlog.ip = ip
            auditlog.save()

class Migration(migrations.Migration):
    dependencies = [('audit', '0004_change_ip_field_type')]
    operations = [migrations.RunPython(forwards_func)]