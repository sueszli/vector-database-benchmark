from google.cloud import documentai_v1beta3

def sample_import_documents():
    if False:
        i = 10
        return i + 15
    client = documentai_v1beta3.DocumentServiceClient()
    batch_documents_import_configs = documentai_v1beta3.BatchDocumentsImportConfig()
    batch_documents_import_configs.dataset_split = 'DATASET_SPLIT_UNASSIGNED'
    request = documentai_v1beta3.ImportDocumentsRequest(dataset='dataset_value', batch_documents_import_configs=batch_documents_import_configs)
    operation = client.import_documents(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)