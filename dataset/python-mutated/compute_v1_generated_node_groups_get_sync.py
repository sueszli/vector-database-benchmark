from google.cloud import compute_v1

def sample_get():
    if False:
        return 10
    client = compute_v1.NodeGroupsClient()
    request = compute_v1.GetNodeGroupRequest(node_group='node_group_value', project='project_value', zone='zone_value')
    response = client.get(request=request)
    print(response)