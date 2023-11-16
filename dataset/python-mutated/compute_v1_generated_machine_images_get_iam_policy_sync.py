from google.cloud import compute_v1

def sample_get_iam_policy():
    if False:
        print('Hello World!')
    client = compute_v1.MachineImagesClient()
    request = compute_v1.GetIamPolicyMachineImageRequest(project='project_value', resource='resource_value')
    response = client.get_iam_policy(request=request)
    print(response)