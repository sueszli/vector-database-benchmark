from google.cloud import dataform_v1beta1

def sample_get_release_config():
    if False:
        i = 10
        return i + 15
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.GetReleaseConfigRequest(name='name_value')
    response = client.get_release_config(request=request)
    print(response)