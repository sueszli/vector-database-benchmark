from google.cloud import compute_v1

def sample_get_iam_policy():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.NetworkAttachmentsClient()
    request = compute_v1.GetIamPolicyNetworkAttachmentRequest(project='project_value', region='region_value', resource='resource_value')
    response = client.get_iam_policy(request=request)
    print(response)