from google.cloud import workstations_v1

def sample_list_workstation_configs():
    if False:
        for i in range(10):
            print('nop')
    client = workstations_v1.WorkstationsClient()
    request = workstations_v1.ListWorkstationConfigsRequest(parent='parent_value')
    page_result = client.list_workstation_configs(request=request)
    for response in page_result:
        print(response)