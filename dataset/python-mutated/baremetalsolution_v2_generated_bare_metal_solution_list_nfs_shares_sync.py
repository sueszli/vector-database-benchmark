from google.cloud import bare_metal_solution_v2

def sample_list_nfs_shares():
    if False:
        i = 10
        return i + 15
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.ListNfsSharesRequest(parent='parent_value')
    page_result = client.list_nfs_shares(request=request)
    for response in page_result:
        print(response)