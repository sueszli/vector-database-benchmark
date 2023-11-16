from google.cloud import metastore_v1alpha

def sample_list_federations():
    if False:
        for i in range(10):
            print('nop')
    client = metastore_v1alpha.DataprocMetastoreFederationClient()
    request = metastore_v1alpha.ListFederationsRequest(parent='parent_value')
    page_result = client.list_federations(request=request)
    for response in page_result:
        print(response)