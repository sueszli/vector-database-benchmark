from google.cloud import filestore_v1

def sample_list_snapshots():
    if False:
        return 10
    client = filestore_v1.CloudFilestoreManagerClient()
    request = filestore_v1.ListSnapshotsRequest(parent='parent_value')
    page_result = client.list_snapshots(request=request)
    for response in page_result:
        print(response)