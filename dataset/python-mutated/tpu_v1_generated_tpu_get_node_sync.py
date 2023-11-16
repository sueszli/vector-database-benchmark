from google.cloud import tpu_v1

def sample_get_node():
    if False:
        print('Hello World!')
    client = tpu_v1.TpuClient()
    request = tpu_v1.GetNodeRequest(name='name_value')
    response = client.get_node(request=request)
    print(response)