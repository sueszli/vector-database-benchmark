import django.db.models.deletion
from django.db import migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def set_realm_for_existing_scheduledemails(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        while True:
            i = 10
    scheduledemail_model = apps.get_model('zerver', 'ScheduledEmail')
    preregistrationuser_model = apps.get_model('zerver', 'PreregistrationUser')
    for scheduledemail in scheduledemail_model.objects.all():
        if scheduledemail.type == 3:
            prereg = preregistrationuser_model.objects.filter(email=scheduledemail.address).first()
            if prereg is not None:
                scheduledemail.realm = prereg.realm
        else:
            scheduledemail.realm = scheduledemail.user.realm
        scheduledemail.save(update_fields=['realm'])
    scheduledemail_model.objects.filter(realm=None).delete()

class Migration(migrations.Migration):
    dependencies = [('zerver', '0127_disallow_chars_in_stream_and_user_name')]
    operations = [migrations.AddField(model_name='scheduledemail', name='realm', field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='zerver.Realm')), migrations.RunPython(set_realm_for_existing_scheduledemails, reverse_code=migrations.RunPython.noop, elidable=True), migrations.AlterField(model_name='scheduledemail', name='realm', field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='zerver.Realm'))]