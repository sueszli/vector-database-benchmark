from google.cloud import filestore_v1

def sample_delete_snapshot():
    if False:
        print('Hello World!')
    client = filestore_v1.CloudFilestoreManagerClient()
    request = filestore_v1.DeleteSnapshotRequest(name='name_value')
    operation = client.delete_snapshot(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)