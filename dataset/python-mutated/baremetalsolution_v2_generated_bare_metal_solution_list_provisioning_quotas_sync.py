from google.cloud import bare_metal_solution_v2

def sample_list_provisioning_quotas():
    if False:
        return 10
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.ListProvisioningQuotasRequest(parent='parent_value')
    page_result = client.list_provisioning_quotas(request=request)
    for response in page_result:
        print(response)