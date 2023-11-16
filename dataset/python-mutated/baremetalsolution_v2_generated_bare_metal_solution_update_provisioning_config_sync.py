from google.cloud import bare_metal_solution_v2

def sample_update_provisioning_config():
    if False:
        while True:
            i = 10
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.UpdateProvisioningConfigRequest()
    response = client.update_provisioning_config(request=request)
    print(response)