from google.cloud import tpu_v1

def sample_list_accelerator_types():
    if False:
        while True:
            i = 10
    client = tpu_v1.TpuClient()
    request = tpu_v1.ListAcceleratorTypesRequest(parent='parent_value')
    page_result = client.list_accelerator_types(request=request)
    for response in page_result:
        print(response)