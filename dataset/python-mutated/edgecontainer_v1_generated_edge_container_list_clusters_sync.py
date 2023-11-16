from google.cloud import edgecontainer_v1

def sample_list_clusters():
    if False:
        print('Hello World!')
    client = edgecontainer_v1.EdgeContainerClient()
    request = edgecontainer_v1.ListClustersRequest(parent='parent_value')
    page_result = client.list_clusters(request=request)
    for response in page_result:
        print(response)