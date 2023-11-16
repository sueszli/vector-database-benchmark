from google.cloud import bare_metal_solution_v2

def sample_delete_ssh_key():
    if False:
        i = 10
        return i + 15
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.DeleteSSHKeyRequest(name='name_value')
    client.delete_ssh_key(request=request)