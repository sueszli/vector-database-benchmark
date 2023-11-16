from google.cloud import migrationcenter_v1

def sample_get_import_job():
    if False:
        return 10
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.GetImportJobRequest(name='name_value')
    response = client.get_import_job(request=request)
    print(response)