from django.db import connection, migrations, transaction
PRODUCT_BATCH_SIZE = 1000

def data_migration(apps, _schema_editor):
    if False:
        i = 10
        return i + 15
    AssignedProductAttributeValue = apps.get_model('attribute', 'AssignedProductAttributeValue')
    while AssignedProductAttributeValue.objects.filter(product__isnull=True).values_list('pk', flat=True).exists():
        update_product_assignment()

def update_product_assignment():
    if False:
        print('Hello World!')
    'Assign product_id to a new field on assignedproductattributevalue.\n\n    Take the values from attribute_assignedproductattribute to product_id.\n    The old field has already been deleted in Django State operations so we need\n    to use raw SQL to get the value and copy the assignment from the old table.\n    '
    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute('\n                UPDATE attribute_assignedproductattributevalue\n                SET product_id = (\n                    SELECT product_id\n                    FROM attribute_assignedproductattribute\n                    WHERE attribute_assignedproductattributevalue.assignment_id = attribute_assignedproductattribute.id\n                )\n                WHERE id IN (\n                    SELECT ID FROM attribute_assignedproductattributevalue\n                    WHERE product_id IS NULL\n                    ORDER BY ID DESC\n                    FOR UPDATE\n                    LIMIT %s\n                );\n                ', [PRODUCT_BATCH_SIZE])

class Migration(migrations.Migration):
    dependencies = [('attribute', '0035_assignedproductattributevalue_product_add_index')]
    operations = [migrations.RunPython(data_migration, migrations.RunPython.noop)]