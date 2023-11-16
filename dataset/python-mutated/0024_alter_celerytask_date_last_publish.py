import django.db.models
from django.db import migrations, models

def compatible(apps, schema_editor):
    if False:
        return 10
    '\n    兼容旧版本的数据\n    '
    model = apps.get_model('ops', 'JobExecution')
    for obj in model.objects.all():
        if obj.job:
            if obj.job.type == 'adhoc':
                obj.material = '{}:{}'.format(obj.job.module, obj.job.args)
            if obj.job.type == 'playbook':
                obj.material = '{}:{}:{}'.format(obj.job.org.name, obj.job.creator.name, obj.job.playbook.name)
            obj.job_type = obj.job.type
            obj.save()
        else:
            obj.delete()

class Migration(migrations.Migration):
    dependencies = [('ops', '0023_auto_20220912_0021')]
    operations = [migrations.AlterField(model_name='celerytask', name='date_last_publish', field=models.DateTimeField(null=True, verbose_name='Date last publish')), migrations.AlterField(model_name='celerytaskexecution', name='name', field=models.CharField(max_length=1024, verbose_name='Name')), migrations.AddField(model_name='playbook', name='create_method', field=models.CharField(choices=[('blank', 'Blank'), ('vcs', 'VCS')], default='blank', max_length=128, verbose_name='CreateMethod')), migrations.AddField(model_name='playbook', name='vcs_url', field=models.CharField(blank=True, default='', max_length=1024, null=True, verbose_name='VCS URL')), migrations.AlterField(model_name='jobexecution', name='job', field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='executions', to='ops.job')), migrations.AddField(model_name='jobexecution', name='job_type', field=models.CharField(choices=[('adhoc', 'Adhoc'), ('playbook', 'Playbook')], default='adhoc', max_length=128, verbose_name='Material Type')), migrations.AddField(model_name='jobexecution', name='material', field=models.CharField(blank=True, default='', max_length=1024, null=True, verbose_name='Material')), migrations.DeleteModel(name='JobAuditLog'), migrations.RunPython(compatible)]