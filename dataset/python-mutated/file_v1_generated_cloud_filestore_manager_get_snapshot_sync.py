from google.cloud import filestore_v1

def sample_get_snapshot():
    if False:
        print('Hello World!')
    client = filestore_v1.CloudFilestoreManagerClient()
    request = filestore_v1.GetSnapshotRequest(name='name_value')
    response = client.get_snapshot(request=request)
    print(response)