from google.cloud import tpu_v2alpha1

def sample_get_guest_attributes():
    if False:
        print('Hello World!')
    client = tpu_v2alpha1.TpuClient()
    request = tpu_v2alpha1.GetGuestAttributesRequest(name='name_value')
    response = client.get_guest_attributes(request=request)
    print(response)