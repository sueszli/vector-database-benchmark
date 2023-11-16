from django.db import migrations, models, connection

def migrate_event_data(apps, schema_editor):
    if False:
        while True:
            i = 10
    for tblname in ('main_jobevent', 'main_inventoryupdateevent', 'main_projectupdateevent', 'main_adhoccommandevent', 'main_systemjobevent'):
        with connection.cursor() as cursor:
            cursor.execute(f'ALTER TABLE {tblname} ALTER COLUMN id TYPE bigint USING id::bigint;')

class FakeAlterField(migrations.AlterField):

    def database_forwards(self, *args):
        if False:
            while True:
                i = 10
        pass

class Migration(migrations.Migration):
    dependencies = [('main', '0112_v370_workflow_node_identifier')]
    operations = [migrations.RunPython(migrate_event_data), FakeAlterField(model_name='adhoccommandevent', name='id', field=models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')), FakeAlterField(model_name='inventoryupdateevent', name='id', field=models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')), FakeAlterField(model_name='jobevent', name='id', field=models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')), FakeAlterField(model_name='projectupdateevent', name='id', field=models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')), FakeAlterField(model_name='systemjobevent', name='id', field=models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID'))]