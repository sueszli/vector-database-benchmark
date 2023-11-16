from google.cloud import tpu_v2

def sample_get_guest_attributes():
    if False:
        for i in range(10):
            print('nop')
    client = tpu_v2.TpuClient()
    request = tpu_v2.GetGuestAttributesRequest(name='name_value')
    response = client.get_guest_attributes(request=request)
    print(response)