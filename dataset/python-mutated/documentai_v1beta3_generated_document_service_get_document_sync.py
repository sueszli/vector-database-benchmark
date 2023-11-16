from google.cloud import documentai_v1beta3

def sample_get_document():
    if False:
        return 10
    client = documentai_v1beta3.DocumentServiceClient()
    document_id = documentai_v1beta3.DocumentId()
    document_id.gcs_managed_doc_id.gcs_uri = 'gcs_uri_value'
    request = documentai_v1beta3.GetDocumentRequest(dataset='dataset_value', document_id=document_id)
    response = client.get_document(request=request)
    print(response)