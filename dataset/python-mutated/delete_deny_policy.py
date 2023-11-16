def delete_deny_policy(project_id: str, policy_id: str) -> None:
    if False:
        while True:
            i = 10
    from google.cloud import iam_v2
    from google.cloud.iam_v2 import types
    '\n    Delete the policy if you no longer want to enforce the rules in a deny policy.\n\n    project_id: ID or number of the Google Cloud project you want to use.\n    policy_id: The ID of the deny policy you want to retrieve.\n    '
    policies_client = iam_v2.PoliciesClient()
    attachment_point = f'cloudresourcemanager.googleapis.com%2Fprojects%2F{project_id}'
    request = types.DeletePolicyRequest()
    request.name = f'policies/{attachment_point}/denypolicies/{policy_id}'
    result = policies_client.delete_policy(request=request).result()
    print(f"Deleted the deny policy: {result.name.rsplit('/')[-1]}")
if __name__ == '__main__':
    import uuid
    project_id = 'your-google-cloud-project-id'
    policy_id = f'deny-{uuid.uuid4()}'
    delete_deny_policy(project_id, policy_id)