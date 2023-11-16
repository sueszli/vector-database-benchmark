from google.cloud import clouddms_v1

def sample_import_mapping_rules():
    if False:
        print('Hello World!')
    client = clouddms_v1.DataMigrationServiceClient()
    rules_files = clouddms_v1.RulesFile()
    rules_files.rules_source_filename = 'rules_source_filename_value'
    rules_files.rules_content = 'rules_content_value'
    request = clouddms_v1.ImportMappingRulesRequest(parent='parent_value', rules_format='IMPORT_RULES_FILE_FORMAT_ORATOPG_CONFIG_FILE', rules_files=rules_files, auto_commit=True)
    operation = client.import_mapping_rules(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)