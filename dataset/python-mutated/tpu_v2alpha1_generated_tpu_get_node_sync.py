from google.cloud import tpu_v2alpha1

def sample_get_node():
    if False:
        for i in range(10):
            print('nop')
    client = tpu_v2alpha1.TpuClient()
    request = tpu_v2alpha1.GetNodeRequest(name='name_value')
    response = client.get_node(request=request)
    print(response)