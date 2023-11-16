"""This application demonstrates how to perform create operations
on Product set in Cloud Vision Product Search.

For more information, see the tutorial page at
https://cloud.google.com/vision/product-search/docs/
"""
import argparse
from google.cloud import vision

def add_product_to_product_set(project_id, location, product_id, product_set_id):
    if False:
        return 10
    'Add a product to a product set.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n        product_id: Id of the product.\n        product_set_id: Id of the product set.\n    '
    client = vision.ProductSearchClient()
    product_set_path = client.product_set_path(project=project_id, location=location, product_set=product_set_id)
    product_path = client.product_path(project=project_id, location=location, product=product_id)
    client.add_product_to_product_set(name=product_set_path, product=product_path)
    print('Product added to product set.')

def list_products_in_product_set(project_id, location, product_set_id):
    if False:
        i = 10
        return i + 15
    'List all products in a product set.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n        product_set_id: Id of the product set.\n    '
    client = vision.ProductSearchClient()
    product_set_path = client.product_set_path(project=project_id, location=location, product_set=product_set_id)
    products = client.list_products_in_product_set(name=product_set_path)
    for product in products:
        print(f'Product name: {product.name}')
        print('Product id: {}'.format(product.name.split('/')[-1]))
        print(f'Product display name: {product.display_name}')
        print(f'Product description: {product.description}')
        print(f'Product category: {product.product_category}')
        print(f'Product labels: {product.product_labels}')

def remove_product_from_product_set(project_id, location, product_id, product_set_id):
    if False:
        for i in range(10):
            print('nop')
    'Remove a product from a product set.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n        product_id: Id of the product.\n        product_set_id: Id of the product set.\n    '
    client = vision.ProductSearchClient()
    product_set_path = client.product_set_path(project=project_id, location=location, product_set=product_set_id)
    product_path = client.product_path(project=project_id, location=location, product=product_id)
    client.remove_product_from_product_set(name=product_set_path, product=product_path)
    print('Product removed from product set.')

def purge_products_in_product_set(project_id, location, product_set_id, force):
    if False:
        for i in range(10):
            print('nop')
    'Delete all products in a product set.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n        product_set_id: Id of the product set.\n        force: Perform the purge only when force is set to True.\n    '
    client = vision.ProductSearchClient()
    parent = f'projects/{project_id}/locations/{location}'
    product_set_purge_config = vision.ProductSetPurgeConfig(product_set_id=product_set_id)
    operation = client.purge_products(request={'parent': parent, 'product_set_purge_config': product_set_purge_config, 'force': force})
    operation.result(timeout=500)
    print('Deleted products in product set.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    parser.add_argument('--project_id', help='Project id.  Required', required=True)
    parser.add_argument('--location', help='Compute region name', default='us-west1')
    add_product_to_product_set_parser = subparsers.add_parser('add_product_to_product_set', help=add_product_to_product_set.__doc__)
    add_product_to_product_set_parser.add_argument('product_id')
    add_product_to_product_set_parser.add_argument('product_set_id')
    list_products_in_product_set_parser = subparsers.add_parser('list_products_in_product_set', help=list_products_in_product_set.__doc__)
    list_products_in_product_set_parser.add_argument('product_set_id')
    remove_product_from_product_set_parser = subparsers.add_parser('remove_product_from_product_set', help=remove_product_from_product_set.__doc__)
    remove_product_from_product_set_parser.add_argument('product_id')
    remove_product_from_product_set_parser.add_argument('product_set_id')
    purge_products_in_product_set_parser = subparsers.add_parser('purge_products_in_product_set', help=purge_products_in_product_set.__doc__)
    purge_products_in_product_set_parser.add_argument('product_set_id')
    purge_products_in_product_set_parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.command == 'add_product_to_product_set':
        add_product_to_product_set(args.project_id, args.location, args.product_id, args.product_set_id)
    elif args.command == 'list_products_in_product_set':
        list_products_in_product_set(args.project_id, args.location, args.product_set_id)
    elif args.command == 'remove_product_from_product_set':
        remove_product_from_product_set(args.project_id, args.location, args.product_id, args.product_set_id)
    elif args.command == 'purge_products_in_product_set':
        purge_products_in_product_set(args.project_id, args.location, args.product_set_id, args.force)