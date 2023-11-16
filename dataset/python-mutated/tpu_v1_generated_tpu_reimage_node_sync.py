from google.cloud import tpu_v1

def sample_reimage_node():
    if False:
        return 10
    client = tpu_v1.TpuClient()
    request = tpu_v1.ReimageNodeRequest()
    operation = client.reimage_node(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)