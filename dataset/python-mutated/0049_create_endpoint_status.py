from django.db import migrations, models

class Migration(migrations.Migration):
    """
    This script will create endpoint status objects for findings and endpoints for
    databases that already contain those objects.
    """

    def create_status_objects(apps, schema_editor):
        if False:
            return 10
        Finding = apps.get_model('dojo', 'Finding')
        Endpoint_Status = apps.get_model('dojo', 'Endpoint_Status')
        findings = Finding.objects.annotate(count=models.Count('endpoints')).filter(count__gt=0)
        for finding in findings:
            endpoints = finding.endpoints.all()
            for endpoint in endpoints:
                try:
                    (status, created) = Endpoint_Status.objects.get_or_create(finding=finding, endpoint=endpoint)
                    if created:
                        status.date = finding.date
                        if endpoint.mitigated:
                            status.mitigated = True
                            status.mitigated_by = finding.reporter
                        status.save()
                        endpoint.endpoint_status.add(status)
                        finding.endpoint_status.add(status)
                except Exception as e:
                    print(e)
                    pass
    dependencies = [('dojo', '0048_sla_notifications')]
    operations = [migrations.RunPython(create_status_objects)]