from google.cloud import compute_v1

def sample_set_node_template():
    if False:
        print('Hello World!')
    client = compute_v1.NodeGroupsClient()
    request = compute_v1.SetNodeTemplateNodeGroupRequest(node_group='node_group_value', project='project_value', zone='zone_value')
    response = client.set_node_template(request=request)
    print(response)