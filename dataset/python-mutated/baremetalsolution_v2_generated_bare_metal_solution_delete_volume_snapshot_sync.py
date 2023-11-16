from google.cloud import bare_metal_solution_v2

def sample_delete_volume_snapshot():
    if False:
        return 10
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.DeleteVolumeSnapshotRequest(name='name_value')
    client.delete_volume_snapshot(request=request)