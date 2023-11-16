from google.cloud import migrationcenter_v1

def sample_create_import_job():
    if False:
        for i in range(10):
            print('nop')
    client = migrationcenter_v1.MigrationCenterClient()
    import_job = migrationcenter_v1.ImportJob()
    import_job.asset_source = 'asset_source_value'
    request = migrationcenter_v1.CreateImportJobRequest(parent='parent_value', import_job_id='import_job_id_value', import_job=import_job)
    operation = client.create_import_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)