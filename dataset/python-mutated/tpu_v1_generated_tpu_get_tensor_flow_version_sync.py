from google.cloud import tpu_v1

def sample_get_tensor_flow_version():
    if False:
        print('Hello World!')
    client = tpu_v1.TpuClient()
    request = tpu_v1.GetTensorFlowVersionRequest(name='name_value')
    response = client.get_tensor_flow_version(request=request)
    print(response)