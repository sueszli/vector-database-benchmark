from google.cloud import compute_v1

def sample_delete():
    if False:
        while True:
            i = 10
    client = compute_v1.RegionDisksClient()
    request = compute_v1.DeleteRegionDiskRequest(disk='disk_value', project='project_value', region='region_value')
    response = client.delete(request=request)
    print(response)