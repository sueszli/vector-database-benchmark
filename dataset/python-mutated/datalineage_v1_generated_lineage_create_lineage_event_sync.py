from google.cloud import datacatalog_lineage_v1

def sample_create_lineage_event():
    if False:
        while True:
            i = 10
    client = datacatalog_lineage_v1.LineageClient()
    request = datacatalog_lineage_v1.CreateLineageEventRequest(parent='parent_value')
    response = client.create_lineage_event(request=request)
    print(response)