from google.cloud import workstations_v1beta

def sample_list_workstations():
    if False:
        for i in range(10):
            print('nop')
    client = workstations_v1beta.WorkstationsClient()
    request = workstations_v1beta.ListWorkstationsRequest(parent='parent_value')
    page_result = client.list_workstations(request=request)
    for response in page_result:
        print(response)