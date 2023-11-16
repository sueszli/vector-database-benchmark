from google.cloud import dataplex_v1

def sample_update_content():
    if False:
        return 10
    client = dataplex_v1.ContentServiceClient()
    content = dataplex_v1.Content()
    content.data_text = 'data_text_value'
    content.sql_script.engine = 'SPARK'
    content.path = 'path_value'
    request = dataplex_v1.UpdateContentRequest(content=content)
    response = client.update_content(request=request)
    print(response)