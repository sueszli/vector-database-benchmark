from google.cloud import compute_v1

def sample_get():
    if False:
        i = 10
        return i + 15
    client = compute_v1.DiskTypesClient()
    request = compute_v1.GetDiskTypeRequest(disk_type='disk_type_value', project='project_value', zone='zone_value')
    response = client.get(request=request)
    print(response)