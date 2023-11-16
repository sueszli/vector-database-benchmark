from google.cloud import tpu_v2alpha1

def sample_reset_queued_resource():
    if False:
        while True:
            i = 10
    client = tpu_v2alpha1.TpuClient()
    request = tpu_v2alpha1.ResetQueuedResourceRequest(name='name_value')
    operation = client.reset_queued_resource(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)