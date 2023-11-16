from google.cloud import workstations_v1

def sample_list_workstations():
    if False:
        while True:
            i = 10
    client = workstations_v1.WorkstationsClient()
    request = workstations_v1.ListWorkstationsRequest(parent='parent_value')
    page_result = client.list_workstations(request=request)
    for response in page_result:
        print(response)