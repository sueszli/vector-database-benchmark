from google.cloud import tpu_v1

def sample_get_accelerator_type():
    if False:
        while True:
            i = 10
    client = tpu_v1.TpuClient()
    request = tpu_v1.GetAcceleratorTypeRequest(name='name_value')
    response = client.get_accelerator_type(request=request)
    print(response)