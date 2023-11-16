import argparse
import os

def main(bucket_name):
    if False:
        print('Hello World!')
    import time
    import google.auth
    from google.cloud.retail import GcsSource, ImportErrorsConfig, ImportUserEventsRequest, UserEventInputConfig, UserEventServiceClient
    project_id = google.auth.default()[1]
    bucket_name = os.getenv('EVENTS_BUCKET_NAME')
    default_catalog = f'projects/{project_id}/locations/global/catalogs/default_catalog'
    gcs_bucket = f'gs://{bucket_name}'
    gcs_errors_bucket = f'{gcs_bucket}/error'
    gcs_events_object = 'user_events.json'

    def get_import_events_gcs_request(gcs_object_name: str):
        if False:
            while True:
                i = 10
        gcs_source = GcsSource()
        gcs_source.input_uris = [f'{gcs_bucket}/{gcs_object_name}']
        input_config = UserEventInputConfig()
        input_config.gcs_source = gcs_source
        errors_config = ImportErrorsConfig()
        errors_config.gcs_prefix = gcs_errors_bucket
        import_request = ImportUserEventsRequest()
        import_request.parent = default_catalog
        import_request.input_config = input_config
        import_request.errors_config = errors_config
        print('---import user events from google cloud source request---')
        print(import_request)
        return import_request

    def import_user_events_from_gcs():
        if False:
            return 10
        import_gcs_request = get_import_events_gcs_request(gcs_events_object)
        gcs_operation = UserEventServiceClient().import_user_events(import_gcs_request)
        print('---the operation was started:----')
        print(gcs_operation.operation.name)
        while not gcs_operation.done():
            print('---please wait till operation is done---')
            time.sleep(30)
        print('---import user events operation is done---')
        if gcs_operation.metadata is not None:
            print('---number of successfully imported events---')
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
    import_user_events_from_gcs()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bucket_name', nargs='?', default=os.environ['EVENTS_BUCKET_NAME'])
    args = parser.parse_args()
    main(args.bucket_name)