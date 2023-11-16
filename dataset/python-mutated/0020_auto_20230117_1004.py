import common.db.encoder
from django.db import migrations, models
from audits.backends.db import OperateLogStore

def migrate_operate_log_after_before(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    operate_log_model = apps.get_model('audits', 'OperateLog')
    db_alias = schema_editor.connection.alias
    (count, batch_size) = (0, 1000)
    while True:
        operate_logs = []
        queryset = operate_log_model.objects.using(db_alias).all()[count:count + batch_size]
        if not queryset:
            break
        count += len(queryset)
        for inst in queryset:
            (before, after, diff) = (inst.before, inst.after, dict())
            if not any([before, after]):
                continue
            diff = OperateLogStore.convert_before_after_to_diff(before, after)
            inst.diff = diff
            operate_logs.append(inst)
        operate_log_model.objects.bulk_update(operate_logs, ['diff'])

class Migration(migrations.Migration):
    dependencies = [('audits', '0019_alter_operatelog_options')]
    operations = [migrations.AddField(model_name='operatelog', name='diff', field=models.JSONField(default=dict, encoder=common.db.encoder.ModelJSONFieldEncoder, null=True)), migrations.RunPython(migrate_operate_log_after_before), migrations.RemoveField(model_name='operatelog', name='after'), migrations.RemoveField(model_name='operatelog', name='before'), migrations.AlterField(model_name='operatelog', name='resource_id', field=models.CharField(blank=True, db_index=True, default='', max_length=128, verbose_name='Resource'))]