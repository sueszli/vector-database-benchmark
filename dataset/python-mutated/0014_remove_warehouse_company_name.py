from django.db import migrations

def migrate_warehouse_address(apps, _schema_editor):
    if False:
        print('Hello World!')
    Warehouse = apps.get_model('warehouse', 'Warehouse')
    for warehouse in Warehouse.objects.filter(company_name__isnull=False, address__company_name='').iterator():
        address = warehouse.address
        address.company_name = warehouse.company_name
        address.save(update_fields=['company_name'])

class Migration(migrations.Migration):
    dependencies = [('warehouse', '0013_auto_20210308_1135')]
    operations = [migrations.RunPython(migrate_warehouse_address, reverse_code=migrations.RunPython.noop), migrations.RemoveField(model_name='warehouse', name='company_name')]