from google.cloud import iam_v2

def sample_delete_policy():
    if False:
        return 10
    client = iam_v2.PoliciesClient()
    request = iam_v2.DeletePolicyRequest(name='name_value')
    operation = client.delete_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)