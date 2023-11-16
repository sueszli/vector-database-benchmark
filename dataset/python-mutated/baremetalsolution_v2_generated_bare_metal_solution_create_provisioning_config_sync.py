from google.cloud import bare_metal_solution_v2

def sample_create_provisioning_config():
    if False:
        for i in range(10):
            print('nop')
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.CreateProvisioningConfigRequest(parent='parent_value')
    response = client.create_provisioning_config(request=request)
    print(response)