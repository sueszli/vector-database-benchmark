from google.cloud import config_v1

def sample_export_lock_info():
    if False:
        return 10
    client = config_v1.ConfigClient()
    request = config_v1.ExportLockInfoRequest(name='name_value')
    response = client.export_lock_info(request=request)
    print(response)