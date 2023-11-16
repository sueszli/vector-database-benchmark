from google.cloud import contentwarehouse_v1

def sample_update_document():
    if False:
        i = 10
        return i + 15
    client = contentwarehouse_v1.DocumentServiceClient()
    document = contentwarehouse_v1.Document()
    document.plain_text = 'plain_text_value'
    document.raw_document_path = 'raw_document_path_value'
    document.display_name = 'display_name_value'
    request = contentwarehouse_v1.UpdateDocumentRequest(name='name_value', document=document)
    response = client.update_document(request=request)
    print(response)