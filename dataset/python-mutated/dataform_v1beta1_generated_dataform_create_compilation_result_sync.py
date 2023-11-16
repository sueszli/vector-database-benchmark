from google.cloud import dataform_v1beta1

def sample_create_compilation_result():
    if False:
        print('Hello World!')
    client = dataform_v1beta1.DataformClient()
    compilation_result = dataform_v1beta1.CompilationResult()
    compilation_result.git_commitish = 'git_commitish_value'
    request = dataform_v1beta1.CreateCompilationResultRequest(parent='parent_value', compilation_result=compilation_result)
    response = client.create_compilation_result(request=request)
    print(response)