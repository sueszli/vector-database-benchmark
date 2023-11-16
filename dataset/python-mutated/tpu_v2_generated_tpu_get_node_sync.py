from google.cloud import tpu_v2

def sample_get_node():
    if False:
        return 10
    client = tpu_v2.TpuClient()
    request = tpu_v2.GetNodeRequest(name='name_value')
    response = client.get_node(request=request)
    print(response)