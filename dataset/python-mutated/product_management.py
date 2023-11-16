"""This application demonstrates how to perform basic operations on Product
in Cloud Vision Product Search.

For more information, see the tutorial page at
https://cloud.google.com/vision/product-search/docs/
"""
import argparse
from google.cloud import vision
from google.protobuf import field_mask_pb2 as field_mask

def create_product(project_id, location, product_id, product_display_name, product_category):
    if False:
        for i in range(10):
            print('nop')
    'Create one product.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n        product_id: Id of the product.\n        product_display_name: Display name of the product.\n        product_category: Category of the product.\n    '
    client = vision.ProductSearchClient()
    location_path = f'projects/{project_id}/locations/{location}'
    product = vision.Product(display_name=product_display_name, product_category=product_category)
    response = client.create_product(parent=location_path, product=product, product_id=product_id)
    print(f'Product name: {response.name}')

def list_products(project_id, location):
    if False:
        print('Hello World!')
    'List all products.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n    '
    client = vision.ProductSearchClient()
    location_path = f'projects/{project_id}/locations/{location}'
    products = client.list_products(parent=location_path)
    for product in products:
        print(f'Product name: {product.name}')
        print('Product id: {}'.format(product.name.split('/')[-1]))
        print(f'Product display name: {product.display_name}')
        print(f'Product description: {product.description}')
        print(f'Product category: {product.product_category}')
        print(f'Product labels: {product.product_labels}\n')

def get_product(project_id, location, product_id):
    if False:
        i = 10
        return i + 15
    'Get information about a product.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n        product_id: Id of the product.\n    '
    client = vision.ProductSearchClient()
    product_path = client.product_path(project=project_id, location=location, product=product_id)
    product = client.get_product(name=product_path)
    print(f'Product name: {product.name}')
    print('Product id: {}'.format(product.name.split('/')[-1]))
    print(f'Product display name: {product.display_name}')
    print(f'Product description: {product.description}')
    print(f'Product category: {product.product_category}')
    print(f'Product labels: {product.product_labels}')

def update_product_labels(project_id, location, product_id, key, value):
    if False:
        i = 10
        return i + 15
    'Update the product labels.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n        product_id: Id of the product.\n        key: The key of the label.\n        value: The value of the label.\n    '
    client = vision.ProductSearchClient()
    product_path = client.product_path(project=project_id, location=location, product=product_id)
    key_value = vision.Product.KeyValue(key=key, value=value)
    product = vision.Product(name=product_path, product_labels=[key_value])
    update_mask = field_mask.FieldMask(paths=['product_labels'])
    updated_product = client.update_product(product=product, update_mask=update_mask)
    print(f'Product name: {updated_product.name}')
    print(f'Updated product labels: {product.product_labels}')

def delete_product(project_id, location, product_id):
    if False:
        for i in range(10):
            print('nop')
    'Delete the product and all its reference images.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n        product_id: Id of the product.\n    '
    client = vision.ProductSearchClient()
    product_path = client.product_path(project=project_id, location=location, product=product_id)
    client.delete_product(name=product_path)
    print('Product deleted.')

def purge_orphan_products(project_id, location, force):
    if False:
        while True:
            i = 10
    'Delete all products not in any product sets.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n    '
    client = vision.ProductSearchClient()
    parent = f'projects/{project_id}/locations/{location}'
    operation = client.purge_products(request={'parent': parent, 'delete_orphan_products': True, 'force': force})
    operation.result(timeout=500)
    print('Orphan products deleted.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project_id', help='Project id.  Required', required=True)
    parser.add_argument('--location', help='Compute region name', default='us-west1')
    subparsers = parser.add_subparsers(dest='command')
    create_product_parser = subparsers.add_parser('create_product', help=create_product.__doc__)
    create_product_parser.add_argument('product_id')
    create_product_parser.add_argument('product_display_name')
    create_product_parser.add_argument('product_category')
    list_products_parser = subparsers.add_parser('list_products', help=list_products.__doc__)
    get_product_parser = subparsers.add_parser('get_product', help=get_product.__doc__)
    get_product_parser.add_argument('product_id')
    update_product_labels_parser = subparsers.add_parser('update_product_labels', help=update_product_labels.__doc__)
    update_product_labels_parser.add_argument('product_id')
    update_product_labels_parser.add_argument('key')
    update_product_labels_parser.add_argument('value')
    delete_product_parser = subparsers.add_parser('delete_product', help=delete_product.__doc__)
    delete_product_parser.add_argument('product_id')
    purge_orphan_products_parser = subparsers.add_parser('purge_orphan_products', help=purge_orphan_products.__doc__)
    purge_orphan_products_parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.command == 'create_product':
        create_product(args.project_id, args.location, args.product_id, args.product_display_name, args.product_category)
    elif args.command == 'list_products':
        list_products(args.project_id, args.location)
    elif args.command == 'get_product':
        get_product(args.project_id, args.location, args.product_id)
    elif args.command == 'update_product_labels':
        update_product_labels(args.project_id, args.location, args.product_id, args.key, args.value)
    elif args.command == 'delete_product':
        delete_product(args.project_id, args.location, args.product_id)
    elif args.command == 'purge_orphan_products':
        purge_orphan_products(args.project_id, args.location, args.force)