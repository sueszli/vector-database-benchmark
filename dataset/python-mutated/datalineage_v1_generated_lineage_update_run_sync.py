from google.cloud import datacatalog_lineage_v1

def sample_update_run():
    if False:
        return 10
    client = datacatalog_lineage_v1.LineageClient()
    run = datacatalog_lineage_v1.Run()
    run.state = 'ABORTED'
    request = datacatalog_lineage_v1.UpdateRunRequest(run=run)
    response = client.update_run(request=request)
    print(response)