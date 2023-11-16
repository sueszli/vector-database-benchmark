from django.db import migrations, models

def update_variant_selection(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    ALLOWED_IN_VARIANT_SELECTION = ['dropdown', 'boolean', 'swatch']
    AttributeVariant = apps.get_model('attribute', 'AttributeVariant')
    attribute_variants = AttributeVariant.objects.select_related('attribute').filter(attribute__input_type__in=ALLOWED_IN_VARIANT_SELECTION)
    attribute_variants.update(variant_selection=True)

class Migration(migrations.Migration):
    dependencies = [('attribute', '0017_auto_20210811_0701')]
    operations = [migrations.AddField(model_name='attributevariant', name='variant_selection', field=models.BooleanField(default=False)), migrations.RunPython(update_variant_selection, reverse_code=migrations.RunPython.noop)]