from google.cloud import compute_v1

def sample_insert():
    if False:
        while True:
            i = 10
    client = compute_v1.NodeGroupsClient()
    request = compute_v1.InsertNodeGroupRequest(initial_node_count=1911, project='project_value', zone='zone_value')
    response = client.insert(request=request)
    print(response)