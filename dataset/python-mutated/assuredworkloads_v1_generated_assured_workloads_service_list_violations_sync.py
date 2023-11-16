from google.cloud import assuredworkloads_v1

def sample_list_violations():
    if False:
        for i in range(10):
            print('nop')
    client = assuredworkloads_v1.AssuredWorkloadsServiceClient()
    request = assuredworkloads_v1.ListViolationsRequest(parent='parent_value')
    page_result = client.list_violations(request=request)
    for response in page_result:
        print(response)