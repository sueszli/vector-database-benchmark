def create_client_with_endpoint(gcp_project_id):
    if False:
        return 10
    'Create a Tables client with a non-default endpoint.'
    from google.cloud import automl_v1beta1 as automl
    from google.api_core.client_options import ClientOptions
    client_options = ClientOptions(api_endpoint='eu-automl.googleapis.com:443')
    client = automl.TablesClient(project=gcp_project_id, region='eu', client_options=client_options)
    print(client.list_datasets())
    return client