from django.db import migrations

def flatten_model_metadata(model_with_metadata):
    if False:
        print('Hello World!')
    updated_fields = []
    public_meta = model_with_metadata.metadata
    private_meta = model_with_metadata.private_metadata
    if public_meta:
        model_with_metadata.metadata = flatten_metadata(public_meta)
        updated_fields.append('metadata')
    if private_meta:
        model_with_metadata.private_metadata = flatten_metadata(private_meta)
        updated_fields.append('private_metadata')
    if updated_fields:
        model_with_metadata.save(update_fields=updated_fields)

def flatten_metadata(metadata):
    if False:
        print('Hello World!')
    flattened_metadata = {}
    for (_, namespace) in metadata.items():
        for (client_name, client) in namespace.items():
            for (key, value) in client.items():
                flattened_key = client_name + '.' + key
                if flattened_key in flattened_metadata:
                    raise Exception(f'Meta key {flattened_key} is duplicated.')
                flattened_metadata[flattened_key] = value
    return flattened_metadata

def flatten_attributes_metadata(apps, _schema_editor):
    if False:
        for i in range(10):
            print('nop')
    Attribute = apps.get_model('product', 'Attribute')
    for attribute in Attribute.objects.iterator():
        flatten_model_metadata(attribute)

def flatten_categories_metadata(apps, _schema_editor):
    if False:
        return 10
    Category = apps.get_model('product', 'Category')
    for category in Category.objects.iterator():
        flatten_model_metadata(category)

def flatten_checkouts_metadata(apps, _schema_editor):
    if False:
        print('Hello World!')
    Checkout = apps.get_model('checkout', 'Checkout')
    for checkout in Checkout.objects.iterator():
        flatten_model_metadata(checkout)

def flatten_collections_metadata(apps, _schema_editor):
    if False:
        print('Hello World!')
    Collection = apps.get_model('product', 'Collection')
    for collection in Collection.objects.iterator():
        flatten_model_metadata(collection)

def flatten_digital_contents_metadata(apps, _schema_editor):
    if False:
        for i in range(10):
            print('nop')
    DigitalContent = apps.get_model('product', 'DigitalContent')
    for digital_content in DigitalContent.objects.iterator():
        flatten_model_metadata(digital_content)

def flatten_fulfillments_metadata(apps, _schema_editor):
    if False:
        return 10
    Fulfillment = apps.get_model('order', 'Fulfillment')
    for fulfillment in Fulfillment.objects.iterator():
        flatten_model_metadata(fulfillment)

def flatten_orders_metadata(apps, _schema_editor):
    if False:
        while True:
            i = 10
    Order = apps.get_model('order', 'Order')
    for order in Order.objects.iterator():
        flatten_model_metadata(order)

def flatten_products_metadata(apps, _schema_editor):
    if False:
        return 10
    Product = apps.get_model('product', 'Product')
    for product in Product.objects.iterator():
        flatten_model_metadata(product)

def flatten_product_types_metadata(apps, _schema_editor):
    if False:
        while True:
            i = 10
    ProductType = apps.get_model('product', 'ProductType')
    for product_type in ProductType.objects.iterator():
        flatten_model_metadata(product_type)

def flatten_product_variants_metadata(apps, _schema_editor):
    if False:
        while True:
            i = 10
    ProductVariant = apps.get_model('product', 'ProductVariant')
    for product_variant in ProductVariant.objects.iterator():
        flatten_model_metadata(product_variant)

def flatten_service_accounts_metadata(apps, _schema_editor):
    if False:
        return 10
    ServiceAccount = apps.get_model('account', 'ServiceAccount')
    for service_account in ServiceAccount.objects.iterator():
        flatten_model_metadata(service_account)

def flatten_users_metadata(apps, _schema_editor):
    if False:
        print('Hello World!')
    User = apps.get_model('account', 'User')
    for user in User.objects.iterator():
        flatten_model_metadata(user)

class Migration(migrations.Migration):
    dependencies = [('account', '0039_auto_20200221_0257'), ('checkout', '0025_auto_20200221_0257'), ('order', '0078_auto_20200221_0257'), ('product', '0115_auto_20200221_0257')]
    run_before = [('attribute', '0001_initial')]
    operations = [migrations.RunPython(flatten_attributes_metadata), migrations.RunPython(flatten_categories_metadata), migrations.RunPython(flatten_checkouts_metadata), migrations.RunPython(flatten_collections_metadata), migrations.RunPython(flatten_digital_contents_metadata), migrations.RunPython(flatten_fulfillments_metadata), migrations.RunPython(flatten_orders_metadata), migrations.RunPython(flatten_products_metadata), migrations.RunPython(flatten_product_types_metadata), migrations.RunPython(flatten_product_variants_metadata), migrations.RunPython(flatten_service_accounts_metadata), migrations.RunPython(flatten_users_metadata)]