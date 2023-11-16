from google.cloud import compute_v1

def sample_delete_nodes():
    if False:
        return 10
    client = compute_v1.NodeGroupsClient()
    request = compute_v1.DeleteNodesNodeGroupRequest(node_group='node_group_value', project='project_value', zone='zone_value')
    response = client.delete_nodes(request=request)
    print(response)