import django.core.validators
from django.db import migrations, models

def migrate_cmd_filter_priority(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    cmd_filter_rule_model = apps.get_model('assets', 'CommandFilterRule')
    cmd_filter_rules = cmd_filter_rule_model.objects.all()
    for cmd_filter_rule in cmd_filter_rules:
        cmd_filter_rule.priority = 100 - cmd_filter_rule.priority + 1
    cmd_filter_rule_model.objects.bulk_update(cmd_filter_rules, fields=['priority'])

def migrate_system_user_priority(apps, schema_editor):
    if False:
        return 10
    system_user_model = apps.get_model('assets', 'SystemUser')
    system_users = system_user_model.objects.all()
    for system_user in system_users:
        system_user.priority = 100 - system_user.priority + 1
    system_user_model.objects.bulk_update(system_users, fields=['priority'])

class Migration(migrations.Migration):
    dependencies = [('assets', '0066_auto_20210208_1802')]
    operations = [migrations.RunPython(migrate_cmd_filter_priority), migrations.RunPython(migrate_system_user_priority), migrations.AlterModelOptions(name='commandfilterrule', options={'ordering': ('priority', 'action'), 'verbose_name': 'Command filter rule'}), migrations.AlterField(model_name='commandfilterrule', name='priority', field=models.IntegerField(default=50, help_text='1-100, the lower the value will be match first', validators=[django.core.validators.MinValueValidator(1), django.core.validators.MaxValueValidator(100)], verbose_name='Priority')), migrations.AlterField(model_name='systemuser', name='priority', field=models.IntegerField(default=20, help_text='1-100, the lower the value will be match first', validators=[django.core.validators.MinValueValidator(1), django.core.validators.MaxValueValidator(100)], verbose_name='Priority'))]