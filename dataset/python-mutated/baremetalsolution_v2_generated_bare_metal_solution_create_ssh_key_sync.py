from google.cloud import bare_metal_solution_v2

def sample_create_ssh_key():
    if False:
        return 10
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.CreateSSHKeyRequest(parent='parent_value', ssh_key_id='ssh_key_id_value')
    response = client.create_ssh_key(request=request)
    print(response)