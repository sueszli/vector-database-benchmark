from google.cloud import automl_v1beta1

def sample_list_column_specs():
    if False:
        return 10
    client = automl_v1beta1.AutoMlClient()
    request = automl_v1beta1.ListColumnSpecsRequest(parent='parent_value')
    page_result = client.list_column_specs(request=request)
    for response in page_result:
        print(response)