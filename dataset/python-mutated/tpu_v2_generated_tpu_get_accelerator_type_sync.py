from google.cloud import tpu_v2

def sample_get_accelerator_type():
    if False:
        print('Hello World!')
    client = tpu_v2.TpuClient()
    request = tpu_v2.GetAcceleratorTypeRequest(name='name_value')
    response = client.get_accelerator_type(request=request)
    print(response)