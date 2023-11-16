from google.cloud import iam_v2

def sample_update_policy():
    if False:
        print('Hello World!')
    client = iam_v2.PoliciesClient()
    request = iam_v2.UpdatePolicyRequest()
    operation = client.update_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)