from google.cloud import bare_metal_solution_v2

def sample_get_provisioning_config():
    if False:
        while True:
            i = 10
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.GetProvisioningConfigRequest(name='name_value')
    response = client.get_provisioning_config(request=request)
    print(response)