from google.cloud import netapp_v1

def sample_create_snapshot():
    if False:
        print('Hello World!')
    client = netapp_v1.NetAppClient()
    request = netapp_v1.CreateSnapshotRequest(parent='parent_value', snapshot_id='snapshot_id_value')
    operation = client.create_snapshot(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)