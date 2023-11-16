"""This application demonstrates how to perform operations
on Product set in Cloud Vision Product Search.

For more information, see the tutorial page at
https://cloud.google.com/vision/product-search/docs/
"""
import argparse
from google.cloud import vision

def create_product_set(project_id, location, product_set_id, product_set_display_name):
    if False:
        return 10
    'Create a product set.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n        product_set_id: Id of the product set.\n        product_set_display_name: Display name of the product set.\n    '
    client = vision.ProductSearchClient()
    location_path = f'projects/{project_id}/locations/{location}'
    product_set = vision.ProductSet(display_name=product_set_display_name)
    response = client.create_product_set(parent=location_path, product_set=product_set, product_set_id=product_set_id)
    print(f'Product set name: {response.name}')

def list_product_sets(project_id, location):
    if False:
        print('Hello World!')
    'List all product sets.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n    '
    client = vision.ProductSearchClient()
    location_path = f'projects/{project_id}/locations/{location}'
    product_sets = client.list_product_sets(parent=location_path)
    for product_set in product_sets:
        print(f'Product set name: {product_set.name}')
        print('Product set id: {}'.format(product_set.name.split('/')[-1]))
        print(f'Product set display name: {product_set.display_name}')
        print('Product set index time: ')
        print(product_set.index_time)

def get_product_set(project_id, location, product_set_id):
    if False:
        for i in range(10):
            print('nop')
    'Get info about the product set.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n        product_set_id: Id of the product set.\n    '
    client = vision.ProductSearchClient()
    product_set_path = client.product_set_path(project=project_id, location=location, product_set=product_set_id)
    product_set = client.get_product_set(name=product_set_path)
    print(f'Product set name: {product_set.name}')
    print('Product set id: {}'.format(product_set.name.split('/')[-1]))
    print(f'Product set display name: {product_set.display_name}')
    print('Product set index time: ')
    print(product_set.index_time)

def delete_product_set(project_id, location, product_set_id):
    if False:
        i = 10
        return i + 15
    'Delete a product set.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n        product_set_id: Id of the product set.\n    '
    client = vision.ProductSearchClient()
    product_set_path = client.product_set_path(project=project_id, location=location, product_set=product_set_id)
    client.delete_product_set(name=product_set_path)
    print('Product set deleted.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    parser.add_argument('--project_id', help='Project id.  Required', required=True)
    parser.add_argument('--location', help='Compute region name', default='us-west1')
    create_product_set_parser = subparsers.add_parser('create_product_set', help=create_product_set.__doc__)
    create_product_set_parser.add_argument('product_set_id')
    create_product_set_parser.add_argument('product_set_display_name')
    list_product_sets_parser = subparsers.add_parser('list_product_sets', help=list_product_sets.__doc__)
    get_product_set_parser = subparsers.add_parser('get_product_set', help=get_product_set.__doc__)
    get_product_set_parser.add_argument('product_set_id')
    delete_product_set_parser = subparsers.add_parser('delete_product_set', help=delete_product_set.__doc__)
    delete_product_set_parser.add_argument('product_set_id')
    args = parser.parse_args()
    if args.command == 'create_product_set':
        create_product_set(args.project_id, args.location, args.product_set_id, args.product_set_display_name)
    elif args.command == 'list_product_sets':
        list_product_sets(args.project_id, args.location)
    elif args.command == 'get_product_set':
        get_product_set(args.project_id, args.location, args.product_set_id)
    elif args.command == 'delete_product_set':
        delete_product_set(args.project_id, args.location, args.product_set_id)