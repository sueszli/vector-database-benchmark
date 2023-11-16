from google.cloud import bare_metal_solution_v2

def sample_list_instances():
    if False:
        while True:
            i = 10
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.ListInstancesRequest(parent='parent_value')
    page_result = client.list_instances(request=request)
    for response in page_result:
        print(response)