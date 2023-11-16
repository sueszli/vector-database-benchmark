from google.cloud import compute_v1

def sample_set_iam_policy():
    if False:
        i = 10
        return i + 15
    client = compute_v1.ServiceAttachmentsClient()
    request = compute_v1.SetIamPolicyServiceAttachmentRequest(project='project_value', region='region_value', resource='resource_value')
    response = client.set_iam_policy(request=request)
    print(response)