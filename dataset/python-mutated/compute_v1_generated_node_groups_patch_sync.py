from google.cloud import compute_v1

def sample_patch():
    if False:
        while True:
            i = 10
    client = compute_v1.NodeGroupsClient()
    request = compute_v1.PatchNodeGroupRequest(node_group='node_group_value', project='project_value', zone='zone_value')
    response = client.patch(request=request)
    print(response)