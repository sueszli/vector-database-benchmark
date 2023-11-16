from google.cloud import migrationcenter_v1

def sample_run_import_job():
    if False:
        while True:
            i = 10
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.RunImportJobRequest(name='name_value')
    operation = client.run_import_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)