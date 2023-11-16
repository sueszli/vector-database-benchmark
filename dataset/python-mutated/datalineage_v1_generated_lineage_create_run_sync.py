from google.cloud import datacatalog_lineage_v1

def sample_create_run():
    if False:
        print('Hello World!')
    client = datacatalog_lineage_v1.LineageClient()
    run = datacatalog_lineage_v1.Run()
    run.state = 'ABORTED'
    request = datacatalog_lineage_v1.CreateRunRequest(parent='parent_value', run=run)
    response = client.create_run(request=request)
    print(response)