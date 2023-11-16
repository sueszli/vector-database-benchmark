"""This application demonstrates how to perform basic operations on reference
images in Cloud Vision Product Search.

For more information, see the tutorial page at
https://cloud.google.com/vision/product-search/docs/
"""
import argparse
from google.cloud import vision

def create_reference_image(project_id, location, product_id, reference_image_id, gcs_uri):
    if False:
        return 10
    'Create a reference image.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n        product_id: Id of the product.\n        reference_image_id: Id of the reference image.\n        gcs_uri: Google Cloud Storage path of the input image.\n    '
    client = vision.ProductSearchClient()
    product_path = client.product_path(project=project_id, location=location, product=product_id)
    reference_image = vision.ReferenceImage(uri=gcs_uri)
    image = client.create_reference_image(parent=product_path, reference_image=reference_image, reference_image_id=reference_image_id)
    print(f'Reference image name: {image.name}')
    print(f'Reference image uri: {image.uri}')

def list_reference_images(project_id, location, product_id):
    if False:
        i = 10
        return i + 15
    'List all images in a product.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n        product_id: Id of the product.\n    '
    client = vision.ProductSearchClient()
    product_path = client.product_path(project=project_id, location=location, product=product_id)
    reference_images = client.list_reference_images(parent=product_path)
    for image in reference_images:
        print(f'Reference image name: {image.name}')
        print('Reference image id: {}'.format(image.name.split('/')[-1]))
        print(f'Reference image uri: {image.uri}')
        print('Reference image bounding polygons: {}'.format(image.bounding_polys))

def get_reference_image(project_id, location, product_id, reference_image_id):
    if False:
        print('Hello World!')
    'Get info about a reference image.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n        product_id: Id of the product.\n        reference_image_id: Id of the reference image.\n    '
    client = vision.ProductSearchClient()
    reference_image_path = client.reference_image_path(project=project_id, location=location, product=product_id, reference_image=reference_image_id)
    image = client.get_reference_image(name=reference_image_path)
    print(f'Reference image name: {image.name}')
    print('Reference image id: {}'.format(image.name.split('/')[-1]))
    print(f'Reference image uri: {image.uri}')
    print(f'Reference image bounding polygons: {image.bounding_polys}')

def delete_reference_image(project_id, location, product_id, reference_image_id):
    if False:
        for i in range(10):
            print('nop')
    'Delete a reference image.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n        product_id: Id of the product.\n        reference_image_id: Id of the reference image.\n    '
    client = vision.ProductSearchClient()
    reference_image_path = client.reference_image_path(project=project_id, location=location, product=product_id, reference_image=reference_image_id)
    client.delete_reference_image(name=reference_image_path)
    print('Reference image deleted from product.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    parser.add_argument('--project_id', help='Project id.  Required', required=True)
    parser.add_argument('--location', help='Compute region name', default='us-west1')
    create_reference_image_parser = subparsers.add_parser('create_reference_image', help=create_reference_image.__doc__)
    create_reference_image_parser.add_argument('product_id')
    create_reference_image_parser.add_argument('reference_image_id')
    create_reference_image_parser.add_argument('gcs_uri')
    list_reference_images_parser = subparsers.add_parser('list_reference_images', help=list_reference_images.__doc__)
    list_reference_images_parser.add_argument('product_id')
    get_reference_image_parser = subparsers.add_parser('get_reference_image', help=get_reference_image.__doc__)
    get_reference_image_parser.add_argument('product_id')
    get_reference_image_parser.add_argument('reference_image_id')
    delete_reference_image_parser = subparsers.add_parser('delete_reference_image', help=delete_reference_image.__doc__)
    delete_reference_image_parser.add_argument('product_id')
    delete_reference_image_parser.add_argument('reference_image_id')
    args = parser.parse_args()
    if args.command == 'create_reference_image':
        create_reference_image(args.project_id, args.location, args.product_id, args.reference_image_id, args.gcs_uri)
    elif args.command == 'list_reference_images':
        list_reference_images(args.project_id, args.location, args.product_id)
    elif args.command == 'get_reference_image':
        get_reference_image(args.project_id, args.location, args.product_id, args.reference_image_id)
    elif args.command == 'delete_reference_image':
        delete_reference_image(args.project_id, args.location, args.product_id, args.reference_image_id)