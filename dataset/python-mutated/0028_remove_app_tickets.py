from django.db import migrations

def migrate_remove_app_tickets(apps, *args):
    if False:
        for i in range(10):
            print('nop')
    model = apps.get_model('tickets', 'Ticket')
    model.objects.filter(type='apply_application').delete()

class Migration(migrations.Migration):
    dependencies = [('tickets', '0027_alter_applycommandticket_apply_run_account')]
    operations = [migrations.RunPython(migrate_remove_app_tickets)]