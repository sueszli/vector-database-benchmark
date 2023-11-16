from django.db import connection, migrations, transaction
PAGE_BATCH_SIZE = 5000

def data_migration(apps, _schema_editor):
    if False:
        while True:
            i = 10
    AssignedPageAttributeValue = apps.get_model('attribute', 'AssignedPageAttributeValue')
    while AssignedPageAttributeValue.objects.filter(page__isnull=True).values_list('pk', flat=True).exists():
        update_page_assignment()

def update_page_assignment():
    if False:
        print('Hello World!')
    "Update Page assignment.\n\n    Update a batch of 'AssignedPageAttributeValue' rows by setting their 'page' based\n    on their related 'assignment'.\n\n    The number of rows updated in each batch is determined by the BATCH_SIZE.\n    Rows are locked during the update to prevent concurrent modifications.\n    "
    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute('\n                WITH limited AS (\n                SELECT av.id\n                FROM attribute_assignedpageattributevalue AS av\n                WHERE av.page_id IS NULL\n                ORDER BY av.id DESC\n                LIMIT %s\n                FOR UPDATE\n            )\n            UPDATE attribute_assignedpageattributevalue AS av\n            SET page_id = apa.page_id\n            FROM attribute_assignedpageattribute AS apa\n            WHERE av.id IN (SELECT id FROM limited)\n            AND av.assignment_id = apa.id;\n            ', [PAGE_BATCH_SIZE])

class Migration(migrations.Migration):
    dependencies = [('page', '0028_add_default_page_type'), ('attribute', '0033_assignedpageattributevalue_page_add_index')]
    operations = [migrations.RunPython(data_migration, migrations.RunPython.noop)]