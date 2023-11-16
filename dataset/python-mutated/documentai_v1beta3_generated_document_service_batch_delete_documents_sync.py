from google.cloud import documentai_v1beta3

def sample_batch_delete_documents():
    if False:
        i = 10
        return i + 15
    client = documentai_v1beta3.DocumentServiceClient()
    dataset_documents = documentai_v1beta3.BatchDatasetDocuments()
    dataset_documents.individual_document_ids.document_ids.gcs_managed_doc_id.gcs_uri = 'gcs_uri_value'
    request = documentai_v1beta3.BatchDeleteDocumentsRequest(dataset='dataset_value', dataset_documents=dataset_documents)
    operation = client.batch_delete_documents(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)