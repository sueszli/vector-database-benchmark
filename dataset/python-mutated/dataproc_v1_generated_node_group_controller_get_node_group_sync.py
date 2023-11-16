from google.cloud import dataproc_v1

def sample_get_node_group():
    if False:
        i = 10
        return i + 15
    client = dataproc_v1.NodeGroupControllerClient()
    request = dataproc_v1.GetNodeGroupRequest(name='name_value')
    response = client.get_node_group(request=request)
    print(response)