from google.cloud import dataproc_v1

def sample_resize_node_group():
    if False:
        i = 10
        return i + 15
    client = dataproc_v1.NodeGroupControllerClient()
    request = dataproc_v1.ResizeNodeGroupRequest(name='name_value', size=443)
    operation = client.resize_node_group(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)