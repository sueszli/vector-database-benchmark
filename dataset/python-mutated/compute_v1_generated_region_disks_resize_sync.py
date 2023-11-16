from google.cloud import compute_v1

def sample_resize():
    if False:
        i = 10
        return i + 15
    client = compute_v1.RegionDisksClient()
    request = compute_v1.ResizeRegionDiskRequest(disk='disk_value', project='project_value', region='region_value')
    response = client.resize(request=request)
    print(response)