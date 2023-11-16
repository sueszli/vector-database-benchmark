from google.cloud import compute_v1

def sample_delete():
    if False:
        print('Hello World!')
    client = compute_v1.NodeGroupsClient()
    request = compute_v1.DeleteNodeGroupRequest(node_group='node_group_value', project='project_value', zone='zone_value')
    response = client.delete(request=request)
    print(response)