from google.cloud import iam_v2beta

def sample_update_policy():
    if False:
        return 10
    client = iam_v2beta.PoliciesClient()
    request = iam_v2beta.UpdatePolicyRequest()
    operation = client.update_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)