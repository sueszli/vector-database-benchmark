from google.cloud import datacatalog_lineage_v1

def sample_get_process():
    if False:
        while True:
            i = 10
    client = datacatalog_lineage_v1.LineageClient()
    request = datacatalog_lineage_v1.GetProcessRequest(name='name_value')
    response = client.get_process(request=request)
    print(response)