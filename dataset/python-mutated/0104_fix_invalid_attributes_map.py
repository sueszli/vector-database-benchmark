import json
from django.db import migrations

def migrate_attributes_to_list(model_name):
    if False:
        i = 10
        return i + 15
    'Migrate HStore attributes configuration to JSONB with a list of values.'

    def make_migration(apps, schema):
        if False:
            for i in range(10):
                print('nop')
        Model = apps.get_model('product', model_name)
        for instance in Model.objects.all():
            new_attributes = {}
            for (k, v) in instance.attributes.items():
                if isinstance(v, str) and (not v.isnumeric()):
                    loaded = json.loads(v.replace("'", '"'))
                    assert isinstance(loaded, list)
                    assert all([isinstance(v_pk, str) for v_pk in loaded])
                    new_attributes[k] = loaded
                elif not isinstance(v, list):
                    new_attributes[k] = [v]
                else:
                    new_attributes[k] = v
            instance.attributes = new_attributes
            instance.save(update_fields=['attributes'])
    return make_migration

class Migration(migrations.Migration):
    dependencies = [('product', '0103_schema_data_enterprise_grade_attributes')]
    operations = [migrations.RunPython(migrate_attributes_to_list('Product')), migrations.RunPython(migrate_attributes_to_list('ProductVariant'))]