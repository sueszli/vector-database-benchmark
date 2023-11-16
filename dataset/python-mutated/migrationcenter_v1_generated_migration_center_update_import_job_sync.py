from google.cloud import migrationcenter_v1

def sample_update_import_job():
    if False:
        while True:
            i = 10
    client = migrationcenter_v1.MigrationCenterClient()
    import_job = migrationcenter_v1.ImportJob()
    import_job.asset_source = 'asset_source_value'
    request = migrationcenter_v1.UpdateImportJobRequest(import_job=import_job)
    operation = client.update_import_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)