from google.cloud import datacatalog_lineage_v1

def sample_delete_run():
    if False:
        while True:
            i = 10
    client = datacatalog_lineage_v1.LineageClient()
    request = datacatalog_lineage_v1.DeleteRunRequest(name='name_value')
    operation = client.delete_run(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)