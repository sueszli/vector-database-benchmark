"""This tutorial demonstrates how users query the product set with their
own images and find the products similer to the image using the Cloud
Vision Product Search API.

For more information, see the tutorial page at
https://cloud.google.com/vision/product-search/docs/
"""
import argparse
from google.cloud import vision

def get_similar_products_file(project_id, location, product_set_id, product_category, file_path, filter, max_results):
    if False:
        for i in range(10):
            print('nop')
    'Search similar products to image.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n        product_set_id: Id of the product set.\n        product_category: Category of the product.\n        file_path: Local file path of the image to be searched.\n        filter: Condition to be applied on the labels.\n                Example for filter: (color = red OR color = blue) AND style = kids\n                It will search on all products with the following labels:\n                color:red AND style:kids\n                color:blue AND style:kids\n        max_results: The maximum number of results (matches) to return. If omitted, all results are returned.\n    '
    product_search_client = vision.ProductSearchClient()
    image_annotator_client = vision.ImageAnnotatorClient()
    with open(file_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    product_set_path = product_search_client.product_set_path(project=project_id, location=location, product_set=product_set_id)
    product_search_params = vision.ProductSearchParams(product_set=product_set_path, product_categories=[product_category], filter=filter)
    image_context = vision.ImageContext(product_search_params=product_search_params)
    response = image_annotator_client.product_search(image, image_context=image_context, max_results=max_results)
    index_time = response.product_search_results.index_time
    print('Product set index time: ')
    print(index_time)
    results = response.product_search_results.results
    print('Search results:')
    for result in results:
        product = result.product
        print(f'Score(Confidence): {result.score}')
        print(f'Image name: {result.image}')
        print(f'Product name: {product.name}')
        print('Product display name: {}'.format(product.display_name))
        print(f'Product description: {product.description}\n')
        print(f'Product labels: {product.product_labels}\n')

def get_similar_products_uri(project_id, location, product_set_id, product_category, image_uri, filter):
    if False:
        while True:
            i = 10
    'Search similar products to image.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n        product_set_id: Id of the product set.\n        product_category: Category of the product.\n        image_uri: Cloud Storage location of image to be searched.\n        filter: Condition to be applied on the labels.\n        Example for filter: (color = red OR color = blue) AND style = kids\n        It will search on all products with the following labels:\n        color:red AND style:kids\n        color:blue AND style:kids\n    '
    product_search_client = vision.ProductSearchClient()
    image_annotator_client = vision.ImageAnnotatorClient()
    image_source = vision.ImageSource(image_uri=image_uri)
    image = vision.Image(source=image_source)
    product_set_path = product_search_client.product_set_path(project=project_id, location=location, product_set=product_set_id)
    product_search_params = vision.ProductSearchParams(product_set=product_set_path, product_categories=[product_category], filter=filter)
    image_context = vision.ImageContext(product_search_params=product_search_params)
    response = image_annotator_client.product_search(image, image_context=image_context)
    index_time = response.product_search_results.index_time
    print('Product set index time: ')
    print(index_time)
    results = response.product_search_results.results
    print('Search results:')
    for result in results:
        product = result.product
        print(f'Score(Confidence): {result.score}')
        print(f'Image name: {result.image}')
        print(f'Product name: {product.name}')
        print('Product display name: {}'.format(product.display_name))
        print(f'Product description: {product.description}\n')
        print(f'Product labels: {product.product_labels}\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    parser.add_argument('--project_id', help='Project id.  Required', required=True)
    parser.add_argument('--location', help='Compute region name', default='us-west1')
    parser.add_argument('--product_set_id')
    parser.add_argument('--product_category')
    parser.add_argument('--filter', default='')
    parser.add_argument('--max_results', default='')
    get_similar_products_file_parser = subparsers.add_parser('get_similar_products_file', help=get_similar_products_file.__doc__)
    get_similar_products_file_parser.add_argument('--file_path')
    get_similar_products_uri_parser = subparsers.add_parser('get_similar_products_uri', help=get_similar_products_uri.__doc__)
    get_similar_products_uri_parser.add_argument('--image_uri')
    args = parser.parse_args()
    if args.command == 'get_similar_products_file':
        get_similar_products_file(args.project_id, args.location, args.product_set_id, args.product_category, args.file_path, args.filter, args.max_results)
    elif args.command == 'get_similar_products_uri':
        get_similar_products_uri(args.project_id, args.location, args.product_set_id, args.product_category, args.image_uri, args.filter, args.max_results)