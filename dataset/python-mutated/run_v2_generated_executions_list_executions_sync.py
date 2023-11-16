from google.cloud import run_v2

def sample_list_executions():
    if False:
        for i in range(10):
            print('nop')
    client = run_v2.ExecutionsClient()
    request = run_v2.ListExecutionsRequest(parent='parent_value')
    page_result = client.list_executions(request=request)
    for response in page_result:
        print(response)