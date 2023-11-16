from google.cloud import edgecontainer_v1

def sample_update_cluster():
    if False:
        while True:
            i = 10
    client = edgecontainer_v1.EdgeContainerClient()
    request = edgecontainer_v1.UpdateClusterRequest()
    operation = client.update_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)