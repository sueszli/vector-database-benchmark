from google.cloud import dataproc_v1

def sample_create_node_group():
    if False:
        print('Hello World!')
    client = dataproc_v1.NodeGroupControllerClient()
    node_group = dataproc_v1.NodeGroup()
    node_group.roles = ['DRIVER']
    request = dataproc_v1.CreateNodeGroupRequest(parent='parent_value', node_group=node_group)
    operation = client.create_node_group(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)