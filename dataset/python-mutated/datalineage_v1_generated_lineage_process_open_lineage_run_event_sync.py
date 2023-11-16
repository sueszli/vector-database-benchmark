from google.cloud import datacatalog_lineage_v1

def sample_process_open_lineage_run_event():
    if False:
        print('Hello World!')
    client = datacatalog_lineage_v1.LineageClient()
    request = datacatalog_lineage_v1.ProcessOpenLineageRunEventRequest(parent='parent_value')
    response = client.process_open_lineage_run_event(request=request)
    print(response)