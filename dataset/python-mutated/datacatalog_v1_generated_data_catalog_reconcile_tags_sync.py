from google.cloud import datacatalog_v1

def sample_reconcile_tags():
    if False:
        i = 10
        return i + 15
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.ReconcileTagsRequest(parent='parent_value', tag_template='tag_template_value')
    operation = client.reconcile_tags(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)