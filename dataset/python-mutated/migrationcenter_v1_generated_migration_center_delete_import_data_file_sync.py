from google.cloud import migrationcenter_v1

def sample_delete_import_data_file():
    if False:
        return 10
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.DeleteImportDataFileRequest(name='name_value')
    operation = client.delete_import_data_file(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)