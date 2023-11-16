from django.db import migrations, models

def migrate_product_attribute_map_to_m2m(apps, schema):
    if False:
        i = 10
        return i + 15
    'Migrate the JSONB attribute map to a M2M relation.'
    Product = apps.get_model('product', 'Product')
    AssignedProductAttribute = apps.get_model('product', 'AssignedProductAttribute')
    product_qs = Product.objects.prefetch_related('product_type__attributeproduct__attribute__values')
    for product in product_qs:
        attribute_map = product.old_attributes
        if not attribute_map:
            continue
        product_type = product.product_type
        for (attribute_pk, values_pk) in attribute_map.items():
            attribute_rel = product_type.attributeproduct.filter(attribute_id=attribute_pk).first()
            if attribute_rel is None:
                continue
            values = list(attribute_rel.attribute.values.filter(pk__in=values_pk))
            if not values:
                continue
            assignment = AssignedProductAttribute.objects.create(product=product, assignment=attribute_rel)
            assignment.values.set(values)

def migrate_variant_attribute_map_to_m2m(apps, schema):
    if False:
        return 10
    'Migrate the JSONB attribute map to a M2M relation.'
    ProductVariant = apps.get_model('product', 'ProductVariant')
    AssignedVariantAttribute = apps.get_model('product', 'AssignedVariantAttribute')
    variants_qs = ProductVariant.objects.prefetch_related('product__product_type__attributevariant__attribute__values')
    for variant in variants_qs:
        attribute_map = variant.old_attributes
        if not attribute_map:
            continue
        product_type = variant.product.product_type
        for (attribute_pk, values_pk) in attribute_map.items():
            attribute_rel = product_type.attributevariant.filter(attribute_id=attribute_pk).first()
            if attribute_rel is None:
                continue
            values = list(attribute_rel.attribute.values.filter(pk__in=values_pk))
            if not values:
                continue
            assignment = AssignedVariantAttribute.objects.create(variant=variant, assignment=attribute_rel)
            assignment.values.set(values)

class Migration(migrations.Migration):
    dependencies = [('product', '0106_django_prices_2')]
    operations = [migrations.RenameField(model_name='product', old_name='attributes', new_name='old_attributes'), migrations.RenameField(model_name='productvariant', old_name='attributes', new_name='old_attributes'), migrations.CreateModel(name='AssignedProductAttribute', fields=[('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')), ('values', models.ManyToManyField(to='product.AttributeValue')), ('product', models.ForeignKey(on_delete=models.deletion.CASCADE, related_name='attributes', to='product.Product')), ('assignment', models.ForeignKey(on_delete=models.deletion.CASCADE, related_name='productassignments', to='product.AttributeProduct'))], options={'unique_together': {('product', 'assignment')}}), migrations.CreateModel(name='AssignedVariantAttribute', fields=[('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')), ('values', models.ManyToManyField(to='product.AttributeValue')), ('variant', models.ForeignKey(on_delete=models.deletion.CASCADE, related_name='attributes', to='product.ProductVariant')), ('assignment', models.ForeignKey(on_delete=models.deletion.CASCADE, related_name='variantassignments', to='product.AttributeVariant'))], options={'unique_together': {('variant', 'assignment')}}), migrations.AddField(model_name='attributeproduct', name='assigned_products', field=models.ManyToManyField(blank=True, through='product.AssignedProductAttribute', to='product.Product', related_name='attributesrelated')), migrations.AddField(model_name='attributevariant', name='assigned_variants', field=models.ManyToManyField(blank=True, through='product.AssignedVariantAttribute', to='product.ProductVariant', related_name='attributesrelated')), migrations.AlterModelOptions(name='attributeproduct', options={'ordering': ('sort_order',)}), migrations.AlterModelOptions(name='attributevariant', options={'ordering': ('sort_order',)}), migrations.RunPython(migrate_product_attribute_map_to_m2m), migrations.RunPython(migrate_variant_attribute_map_to_m2m), migrations.RemoveField(model_name='product', name='old_attributes'), migrations.RemoveField(model_name='productvariant', name='old_attributes')]