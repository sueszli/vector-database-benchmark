from google.cloud import datacatalog_lineage_v1

def sample_delete_lineage_event():
    if False:
        while True:
            i = 10
    client = datacatalog_lineage_v1.LineageClient()
    request = datacatalog_lineage_v1.DeleteLineageEventRequest(name='name_value')
    client.delete_lineage_event(request=request)