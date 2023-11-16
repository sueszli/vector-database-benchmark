from google.cloud import bare_metal_solution_v2

def sample_submit_provisioning_config():
    if False:
        return 10
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.SubmitProvisioningConfigRequest(parent='parent_value')
    response = client.submit_provisioning_config(request=request)
    print(response)