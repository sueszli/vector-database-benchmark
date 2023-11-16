from google.cloud import dataform_v1beta1

def sample_delete_release_config():
    if False:
        print('Hello World!')
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.DeleteReleaseConfigRequest(name='name_value')
    client.delete_release_config(request=request)