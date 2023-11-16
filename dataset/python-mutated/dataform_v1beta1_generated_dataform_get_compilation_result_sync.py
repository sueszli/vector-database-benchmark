from google.cloud import dataform_v1beta1

def sample_get_compilation_result():
    if False:
        print('Hello World!')
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.GetCompilationResultRequest(name='name_value')
    response = client.get_compilation_result(request=request)
    print(response)