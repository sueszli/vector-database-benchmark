from google.cloud import filestore_v1

def sample_update_snapshot():
    if False:
        while True:
            i = 10
    client = filestore_v1.CloudFilestoreManagerClient()
    request = filestore_v1.UpdateSnapshotRequest()
    operation = client.update_snapshot(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)