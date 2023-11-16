from google.cloud import migrationcenter_v1

def sample_delete_import_job():
    if False:
        return 10
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.DeleteImportJobRequest(name='name_value')
    operation = client.delete_import_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)