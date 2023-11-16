from google.cloud import dataform_v1beta1

def sample_create_repository():
    if False:
        for i in range(10):
            print('nop')
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.CreateRepositoryRequest(parent='parent_value', repository_id='repository_id_value')
    response = client.create_repository(request=request)
    print(response)