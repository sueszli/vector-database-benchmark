from google.cloud import compute_v1

def sample_resize():
    if False:
        while True:
            i = 10
    client = compute_v1.DisksClient()
    request = compute_v1.ResizeDiskRequest(disk='disk_value', project='project_value', zone='zone_value')
    response = client.resize(request=request)
    print(response)