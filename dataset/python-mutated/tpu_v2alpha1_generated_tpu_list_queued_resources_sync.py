from google.cloud import tpu_v2alpha1

def sample_list_queued_resources():
    if False:
        print('Hello World!')
    client = tpu_v2alpha1.TpuClient()
    request = tpu_v2alpha1.ListQueuedResourcesRequest(parent='parent_value')
    page_result = client.list_queued_resources(request=request)
    for response in page_result:
        print(response)