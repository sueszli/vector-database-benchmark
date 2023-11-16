from google.cloud import dataplex_v1

def sample_create_content():
    if False:
        for i in range(10):
            print('nop')
    client = dataplex_v1.ContentServiceClient()
    content = dataplex_v1.Content()
    content.data_text = 'data_text_value'
    content.sql_script.engine = 'SPARK'
    content.path = 'path_value'
    request = dataplex_v1.CreateContentRequest(parent='parent_value', content=content)
    response = client.create_content(request=request)
    print(response)