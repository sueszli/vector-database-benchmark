from google.cloud import bare_metal_solution_v2

def sample_list_volume_snapshots():
    if False:
        return 10
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.ListVolumeSnapshotsRequest(parent='parent_value')
    page_result = client.list_volume_snapshots(request=request)
    for response in page_result:
        print(response)