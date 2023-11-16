from google.cloud import datacatalog_lineage_v1

def sample_update_process():
    if False:
        return 10
    client = datacatalog_lineage_v1.LineageClient()
    request = datacatalog_lineage_v1.UpdateProcessRequest()
    response = client.update_process(request=request)
    print(response)