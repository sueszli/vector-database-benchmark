from google.cloud import bare_metal_solution_v2

def sample_rename_nfs_share():
    if False:
        print('Hello World!')
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.RenameNfsShareRequest(name='name_value', new_nfsshare_id='new_nfsshare_id_value')
    response = client.rename_nfs_share(request=request)
    print(response)