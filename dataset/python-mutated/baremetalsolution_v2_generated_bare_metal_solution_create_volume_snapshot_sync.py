from google.cloud import bare_metal_solution_v2

def sample_create_volume_snapshot():
    if False:
        for i in range(10):
            print('nop')
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.CreateVolumeSnapshotRequest(parent='parent_value')
    response = client.create_volume_snapshot(request=request)
    print(response)