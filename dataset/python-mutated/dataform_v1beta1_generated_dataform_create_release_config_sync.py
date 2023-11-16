from google.cloud import dataform_v1beta1

def sample_create_release_config():
    if False:
        return 10
    client = dataform_v1beta1.DataformClient()
    release_config = dataform_v1beta1.ReleaseConfig()
    release_config.git_commitish = 'git_commitish_value'
    request = dataform_v1beta1.CreateReleaseConfigRequest(parent='parent_value', release_config=release_config, release_config_id='release_config_id_value')
    response = client.create_release_config(request=request)
    print(response)