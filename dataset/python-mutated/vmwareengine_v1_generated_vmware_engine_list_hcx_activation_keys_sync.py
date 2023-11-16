from google.cloud import vmwareengine_v1

def sample_list_hcx_activation_keys():
    if False:
        for i in range(10):
            print('nop')
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.ListHcxActivationKeysRequest(parent='parent_value')
    page_result = client.list_hcx_activation_keys(request=request)
    for response in page_result:
        print(response)