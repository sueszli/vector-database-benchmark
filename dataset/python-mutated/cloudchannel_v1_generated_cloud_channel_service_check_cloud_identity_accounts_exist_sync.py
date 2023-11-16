from google.cloud import channel_v1

def sample_check_cloud_identity_accounts_exist():
    if False:
        i = 10
        return i + 15
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.CheckCloudIdentityAccountsExistRequest(parent='parent_value', domain='domain_value')
    response = client.check_cloud_identity_accounts_exist(request=request)
    print(response)