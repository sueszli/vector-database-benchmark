from google.cloud import datacatalog_lineage_v1

def sample_get_run():
    if False:
        return 10
    client = datacatalog_lineage_v1.LineageClient()
    request = datacatalog_lineage_v1.GetRunRequest(name='name_value')
    response = client.get_run(request=request)
    print(response)