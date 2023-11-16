from google.cloud import datacatalog_lineage_v1

def sample_delete_process():
    if False:
        i = 10
        return i + 15
    client = datacatalog_lineage_v1.LineageClient()
    request = datacatalog_lineage_v1.DeleteProcessRequest(name='name_value')
    operation = client.delete_process(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)