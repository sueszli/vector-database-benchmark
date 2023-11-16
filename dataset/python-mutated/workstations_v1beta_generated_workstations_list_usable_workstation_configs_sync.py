from google.cloud import workstations_v1beta

def sample_list_usable_workstation_configs():
    if False:
        while True:
            i = 10
    client = workstations_v1beta.WorkstationsClient()
    request = workstations_v1beta.ListUsableWorkstationConfigsRequest(parent='parent_value')
    page_result = client.list_usable_workstation_configs(request=request)
    for response in page_result:
        print(response)