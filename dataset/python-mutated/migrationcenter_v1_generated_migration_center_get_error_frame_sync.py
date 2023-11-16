from google.cloud import migrationcenter_v1

def sample_get_error_frame():
    if False:
        print('Hello World!')
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.GetErrorFrameRequest(name='name_value')
    response = client.get_error_frame(request=request)
    print(response)