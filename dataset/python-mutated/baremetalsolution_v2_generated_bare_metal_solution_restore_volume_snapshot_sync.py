from google.cloud import bare_metal_solution_v2

def sample_restore_volume_snapshot():
    if False:
        i = 10
        return i + 15
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.RestoreVolumeSnapshotRequest(volume_snapshot='volume_snapshot_value')
    operation = client.restore_volume_snapshot(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)