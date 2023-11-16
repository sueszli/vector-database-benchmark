from google.cloud import bare_metal_solution_v2

def sample_list_volumes():
    if False:
        print('Hello World!')
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.ListVolumesRequest(parent='parent_value')
    page_result = client.list_volumes(request=request)
    for response in page_result:
        print(response)