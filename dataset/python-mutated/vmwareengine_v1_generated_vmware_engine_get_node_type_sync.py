from google.cloud import vmwareengine_v1

def sample_get_node_type():
    if False:
        for i in range(10):
            print('nop')
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.GetNodeTypeRequest(name='name_value')
    response = client.get_node_type(request=request)
    print(response)