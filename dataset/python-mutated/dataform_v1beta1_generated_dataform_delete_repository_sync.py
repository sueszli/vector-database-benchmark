from google.cloud import dataform_v1beta1

def sample_delete_repository():
    if False:
        while True:
            i = 10
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.DeleteRepositoryRequest(name='name_value')
    client.delete_repository(request=request)