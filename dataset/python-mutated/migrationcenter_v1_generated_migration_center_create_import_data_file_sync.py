from google.cloud import migrationcenter_v1

def sample_create_import_data_file():
    if False:
        print('Hello World!')
    client = migrationcenter_v1.MigrationCenterClient()
    import_data_file = migrationcenter_v1.ImportDataFile()
    import_data_file.format_ = 'IMPORT_JOB_FORMAT_STRATOZONE_CSV'
    request = migrationcenter_v1.CreateImportDataFileRequest(parent='parent_value', import_data_file_id='import_data_file_id_value', import_data_file=import_data_file)
    operation = client.create_import_data_file(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)