from google.cloud import datacatalog_lineage_v1

def sample_create_process():
    if False:
        print('Hello World!')
    client = datacatalog_lineage_v1.LineageClient()
    request = datacatalog_lineage_v1.CreateProcessRequest(parent='parent_value')
    response = client.create_process(request=request)
    print(response)