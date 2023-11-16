from google.cloud import migrationcenter_v1

def sample_get_import_data_file():
    if False:
        for i in range(10):
            print('nop')
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.GetImportDataFileRequest(name='name_value')
    response = client.get_import_data_file(request=request)
    print(response)