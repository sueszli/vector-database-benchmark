from google.cloud import compute_v1

def sample_update():
    if False:
        print('Hello World!')
    client = compute_v1.RegionDisksClient()
    request = compute_v1.UpdateRegionDiskRequest(disk='disk_value', project='project_value', region='region_value')
    response = client.update(request=request)
    print(response)