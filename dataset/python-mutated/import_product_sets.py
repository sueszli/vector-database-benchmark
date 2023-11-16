"""This application demonstrates how to perform import product sets operations
on Product set in Cloud Vision Product Search.

For more information, see the tutorial page at
https://cloud.google.com/vision/product-search/docs/
"""
import argparse
from google.cloud import vision

def import_product_sets(project_id, location, gcs_uri):
    if False:
        while True:
            i = 10
    'Import images of different products in the product set.\n    Args:\n        project_id: Id of the project.\n        location: A compute region name.\n        gcs_uri: Google Cloud Storage URI.\n            Target files must be in Product Search CSV format.\n    '
    client = vision.ProductSearchClient()
    location_path = f'projects/{project_id}/locations/{location}'
    gcs_source = vision.ImportProductSetsGcsSource(csv_file_uri=gcs_uri)
    input_config = vision.ImportProductSetsInputConfig(gcs_source=gcs_source)
    response = client.import_product_sets(parent=location_path, input_config=input_config)
    print(f'Processing operation name: {response.operation.name}')
    result = response.result()
    print('Processing done.')
    for (i, status) in enumerate(result.statuses):
        print('Status of processing line {} of the csv: {}'.format(i, status))
        if status.code == 0:
            reference_image = result.reference_images[i]
            print(reference_image)
        else:
            print(f'Status code not OK: {status.message}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    parser.add_argument('--project_id', help='Project id.  Required', required=True)
    parser.add_argument('--location', help='Compute region name', default='us-west1')
    import_product_sets_parser = subparsers.add_parser('import_product_sets', help=import_product_sets.__doc__)
    import_product_sets_parser.add_argument('gcs_uri')
    args = parser.parse_args()
    if args.command == 'import_product_sets':
        import_product_sets(args.project_id, args.location, args.gcs_uri)