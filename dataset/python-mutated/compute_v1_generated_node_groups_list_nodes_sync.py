from google.cloud import compute_v1

def sample_list_nodes():
    if False:
        return 10
    client = compute_v1.NodeGroupsClient()
    request = compute_v1.ListNodesNodeGroupsRequest(node_group='node_group_value', project='project_value', zone='zone_value')
    page_result = client.list_nodes(request=request)
    for response in page_result:
        print(response)