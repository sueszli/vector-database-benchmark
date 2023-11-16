from google.cloud import compute_v1

def sample_update():
    if False:
        print('Hello World!')
    client = compute_v1.DisksClient()
    request = compute_v1.UpdateDiskRequest(disk='disk_value', project='project_value', zone='zone_value')
    response = client.update(request=request)
    print(response)