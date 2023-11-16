from google.cloud import automl_v1beta1

def sample_list_datasets():
    if False:
        for i in range(10):
            print('nop')
    client = automl_v1beta1.AutoMlClient()
    request = automl_v1beta1.ListDatasetsRequest(parent='parent_value')
    page_result = client.list_datasets(request=request)
    for response in page_result:
        print(response)