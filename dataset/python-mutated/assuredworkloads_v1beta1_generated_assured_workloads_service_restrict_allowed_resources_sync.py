from google.cloud import assuredworkloads_v1beta1

def sample_restrict_allowed_resources():
    if False:
        print('Hello World!')
    client = assuredworkloads_v1beta1.AssuredWorkloadsServiceClient()
    request = assuredworkloads_v1beta1.RestrictAllowedResourcesRequest(name='name_value', restriction_type='ALLOW_COMPLIANT_RESOURCES')
    response = client.restrict_allowed_resources(request=request)
    print(response)