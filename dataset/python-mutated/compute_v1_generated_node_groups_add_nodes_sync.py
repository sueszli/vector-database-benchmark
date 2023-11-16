from google.cloud import compute_v1

def sample_add_nodes():
    if False:
        print('Hello World!')
    client = compute_v1.NodeGroupsClient()
    request = compute_v1.AddNodesNodeGroupRequest(node_group='node_group_value', project='project_value', zone='zone_value')
    response = client.add_nodes(request=request)
    print(response)