import argparse
import os

def main(bucket_name):
    if False:
        for i in range(10):
            print('nop')
    import time
    import google.auth
    from google.cloud.retail import GcsSource, ImportErrorsConfig, ImportProductsRequest, ProductInputConfig, ProductServiceClient
    project_id = google.auth.default()[1]
    default_catalog = f'projects/{project_id}/locations/global/catalogs/default_catalog/branches/default_branch'
    gcs_bucket = f'gs://{bucket_name}'
    gcs_errors_bucket = f'{gcs_bucket}/error'
    gcs_products_object = 'products.json'

    def get_import_products_gcs_request(gcs_object_name: str):
        if False:
            print('Hello World!')
        gcs_source = GcsSource()
        gcs_source.input_uris = [f'{gcs_bucket}/{gcs_object_name}']
        input_config = ProductInputConfig()
        input_config.gcs_source = gcs_source
        print('GRS source:')
        print(gcs_source.input_uris)
        errors_config = ImportErrorsConfig()
        errors_config.gcs_prefix = gcs_errors_bucket
        import_request = ImportProductsRequest()
        import_request.parent = default_catalog
        import_request.reconciliation_mode = ImportProductsRequest.ReconciliationMode.INCREMENTAL
        import_request.input_config = input_config
        import_request.errors_config = errors_config
        print('---import products from google cloud source request---')
        print(import_request)
        return import_request

    def import_products_from_gcs():
        if False:
            while True:
                i = 10
        import_gcs_request = get_import_products_gcs_request(gcs_products_object)
        gcs_operation = ProductServiceClient().import_products(import_gcs_request)
        print('---the operation was started:----')
        print(gcs_operation.operation.name)
        while not gcs_operation.done():
            print('---please wait till operation is done---')
            time.sleep(30)
        print('---import products operation is done---')
        if gcs_operation.metadata is not None:
            print('---number of successfully imported products---')
            print(gcs_operation.metadata.success_count)
            print('---number of failures during the importing---')
            print(gcs_operation.metadata.failure_count)
        else:
            print('---operation.metadata is empty---')
        if gcs_operation.result is not None:
            print('---operation result:---')
            print(gcs_operation.result())
        else:
            print('---operation.result is empty---')
        print('Wait 2-5 minutes till products become indexed in the catalog, after that they will be available for search')
    import_products_from_gcs()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bucket_name', nargs='?', default=os.environ['BUCKET_NAME'])
    args = parser.parse_args()
    main(args.bucket_name)