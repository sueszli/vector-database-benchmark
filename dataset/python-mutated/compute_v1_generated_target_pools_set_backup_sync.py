from google.cloud import compute_v1

def sample_set_backup():
    if False:
        while True:
            i = 10
    client = compute_v1.TargetPoolsClient()
    request = compute_v1.SetBackupTargetPoolRequest(project='project_value', region='region_value', target_pool='target_pool_value')
    response = client.set_backup(request=request)
    print(response)