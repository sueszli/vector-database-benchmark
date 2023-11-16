from google.cloud import dataform_v1beta1

def sample_query_compilation_result_actions():
    if False:
        for i in range(10):
            print('nop')
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.QueryCompilationResultActionsRequest(name='name_value')
    page_result = client.query_compilation_result_actions(request=request)
    for response in page_result:
        print(response)