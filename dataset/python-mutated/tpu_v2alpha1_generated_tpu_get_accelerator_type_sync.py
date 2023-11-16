from google.cloud import tpu_v2alpha1

def sample_get_accelerator_type():
    if False:
        i = 10
        return i + 15
    client = tpu_v2alpha1.TpuClient()
    request = tpu_v2alpha1.GetAcceleratorTypeRequest(name='name_value')
    response = client.get_accelerator_type(request=request)
    print(response)