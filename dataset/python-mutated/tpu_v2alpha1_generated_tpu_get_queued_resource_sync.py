from google.cloud import tpu_v2alpha1

def sample_get_queued_resource():
    if False:
        i = 10
        return i + 15
    client = tpu_v2alpha1.TpuClient()
    request = tpu_v2alpha1.GetQueuedResourceRequest(name='name_value')
    response = client.get_queued_resource(request=request)
    print(response)