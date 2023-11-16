from django.db import migrations

def clone_attribute_data(original_attribute, new_attribute):
    if False:
        i = 10
        return i + 15
    for translation in original_attribute.translations.all():
        new_attribute.translations.create(language_code=translation.language_code, name=translation.name)
    for value in original_attribute.values.all():
        new_value = new_attribute.values.create(name=value.name, value=value.value, slug=value.slug, sort_order=value.sort_order)
        for translation in value.translations.all():
            new_value.translations.create(language_code=translation.language_code, name=translation.name)

def migrate_attributes(apps, schema_editor):
    if False:
        print('Hello World!')
    Attribute = apps.get_model('product', 'Attribute')
    for attribute in Attribute.objects.all():
        for product_type in attribute.product_types.all():
            new_attr = product_type.temp_product_attributes.create(name=attribute.name, slug=attribute.slug)
            clone_attribute_data(attribute, new_attr)
        for product_type in attribute.product_variant_types.all():
            new_attr = product_type.temp_variant_attributes.create(name=attribute.name, slug=attribute.slug)
            clone_attribute_data(attribute, new_attr)

def migrate_attributes_hstore_to_new_ids(apps, schema_editor):
    if False:
        print('Hello World!')
    Product = apps.get_model('product', 'Product')
    Attribute = apps.get_model('product', 'Attribute')
    AttributeValue = apps.get_model('product', 'AttributeValue')
    attributes_map = {str(attr.pk): attr for attr in Attribute.objects.all()}
    values_map = {str(val.pk): val for val in AttributeValue.objects.all()}
    qs = Product.objects.select_related('product_type').prefetch_related('variants')
    for product in qs:
        if product.attributes:
            new_hstore = {}
            product_type = product.product_type
            for (old_attr_pk, old_val_pk) in product.attributes.items():
                old_attr = attributes_map.get(old_attr_pk)
                old_val = values_map.get(old_val_pk)
                if not (old_attr and old_val):
                    continue
                new_attr = product_type.temp_product_attributes.filter(slug=old_attr.slug).first()
                if new_attr:
                    new_val = new_attr.values.filter(slug=old_val.slug).first()
                    if new_val:
                        new_hstore[str(new_attr.pk)] = str(new_val.pk)
            product.attributes = new_hstore
            product.save(update_fields=['attributes'])
        for variant in product.variants.all():
            if variant.attributes:
                new_hstore = {}
                for (old_attr_pk, old_val_pk) in variant.attributes.items():
                    old_attr = attributes_map.get(old_attr_pk)
                    old_val = values_map.get(old_val_pk)
                    if not (old_attr and old_val):
                        continue
                    new_attr = product_type.temp_variant_attributes.filter(slug=old_attr.slug).first()
                    if new_attr:
                        new_val = new_attr.values.filter(slug=old_val.slug).first()
                        if new_val:
                            new_hstore[str(new_attr.pk)] = str(new_val.pk)
                variant.attributes = new_hstore
                variant.save(update_fields=['attributes'])

def clean_stale_attributes(apps, schema_editor):
    if False:
        print('Hello World!')
    Attribute = apps.get_model('product', 'Attribute')
    Attribute.objects.filter(product_variant_type__isnull=True).filter(product_type__isnull=True).delete()

class Migration(migrations.Migration):
    dependencies = [('product', '0073_auto_20181010_0729')]
    operations = [migrations.RunPython(migrate_attributes, reverse_code=migrations.RunPython.noop), migrations.RunPython(migrate_attributes_hstore_to_new_ids, reverse_code=migrations.RunPython.noop), migrations.RunPython(clean_stale_attributes, reverse_code=migrations.RunPython.noop)]