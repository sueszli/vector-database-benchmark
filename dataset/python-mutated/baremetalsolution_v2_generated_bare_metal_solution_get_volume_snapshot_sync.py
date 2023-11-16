from google.cloud import bare_metal_solution_v2

def sample_get_volume_snapshot():
    if False:
        print('Hello World!')
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.GetVolumeSnapshotRequest(name='name_value')
    response = client.get_volume_snapshot(request=request)
    print(response)