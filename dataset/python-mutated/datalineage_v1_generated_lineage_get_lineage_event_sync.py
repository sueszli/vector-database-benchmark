from google.cloud import datacatalog_lineage_v1

def sample_get_lineage_event():
    if False:
        i = 10
        return i + 15
    client = datacatalog_lineage_v1.LineageClient()
    request = datacatalog_lineage_v1.GetLineageEventRequest(name='name_value')
    response = client.get_lineage_event(request=request)
    print(response)