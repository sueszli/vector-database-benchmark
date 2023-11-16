from google.cloud import bare_metal_solution_v2

def sample_list_network_usage():
    if False:
        while True:
            i = 10
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.ListNetworkUsageRequest(location='location_value')
    response = client.list_network_usage(request=request)
    print(response)