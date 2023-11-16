from django.db import migrations, models

def migrate_platform_automation_id(apps, *args):
    if False:
        print('Hello World!')
    platform_model = apps.get_model('assets', 'Platform')
    for platform in platform_model.objects.all():
        if platform.automation:
            platform._automation_id = platform.automation.id
            platform.save(update_fields=['_automation_id'])

def migrate_automation_platform(apps, *args):
    if False:
        i = 10
        return i + 15
    platform_model = apps.get_model('assets', 'Platform')
    automation_model = apps.get_model('assets', 'PlatformAutomation')
    platforms = platform_model.objects.all()
    for platform in platforms:
        if not platform._automation_id:
            continue
        automation = automation_model.objects.filter(id=platform._automation_id).first()
        if not automation:
            continue
        automation.platform = platform
        automation.save(update_fields=['platform'])

class Migration(migrations.Migration):
    dependencies = [('assets', '0114_baseautomation_params')]
    operations = [migrations.AddField(model_name='platform', name='_automation_id', field=models.UUIDField(editable=False, null=True)), migrations.RunPython(migrate_platform_automation_id), migrations.RemoveField(model_name='platform', name='automation'), migrations.AddField(model_name='platformautomation', name='platform', field=models.OneToOneField(null=True, on_delete=models.deletion.CASCADE, related_name='automation', to='assets.platform')), migrations.RunPython(migrate_automation_platform), migrations.RemoveField(model_name='platform', name='_automation_id')]