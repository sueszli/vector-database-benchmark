from google.cloud import netapp_v1

def sample_list_snapshots():
    if False:
        i = 10
        return i + 15
    client = netapp_v1.NetAppClient()
    request = netapp_v1.ListSnapshotsRequest(parent='parent_value')
    page_result = client.list_snapshots(request=request)
    for response in page_result:
        print(response)