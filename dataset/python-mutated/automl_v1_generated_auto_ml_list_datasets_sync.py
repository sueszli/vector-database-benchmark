from google.cloud import automl_v1

def sample_list_datasets():
    if False:
        while True:
            i = 10
    client = automl_v1.AutoMlClient()
    request = automl_v1.ListDatasetsRequest(parent='parent_value')
    page_result = client.list_datasets(request=request)
    for response in page_result:
        print(response)