from google.cloud import edgecontainer_v1

def sample_delete_cluster():
    if False:
        while True:
            i = 10
    client = edgecontainer_v1.EdgeContainerClient()
    request = edgecontainer_v1.DeleteClusterRequest(name='name_value')
    operation = client.delete_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)