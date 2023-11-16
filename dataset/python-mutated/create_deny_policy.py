def create_deny_policy(project_id: str, policy_id: str) -> None:
    if False:
        print('Hello World!')
    from google.cloud import iam_v2
    from google.cloud.iam_v2 import types
    '\n      Create a deny policy.\n      You can add deny policies to organizations, folders, and projects.\n      Each of these resources can have up to 5 deny policies.\n\n      Deny policies contain deny rules, which specify the following:\n      1. The permissions to deny and/or exempt.\n      2. The principals that are denied, or exempted from denial.\n      3. An optional condition on when to enforce the deny rules.\n\n      Params:\n      project_id: ID or number of the Google Cloud project you want to use.\n      policy_id: Specify the ID of the deny policy you want to create.\n    '
    policies_client = iam_v2.PoliciesClient()
    attachment_point = f'cloudresourcemanager.googleapis.com%2Fprojects%2F{project_id}'
    deny_rule = types.DenyRule()
    deny_rule.denied_principals = ['principalSet://goog/public:all']
    deny_rule.denied_permissions = ['cloudresourcemanager.googleapis.com/projects.delete']
    deny_rule.denial_condition = {'expression': "!resource.matchTag('12345678/env', 'test')"}
    policy_rule = types.PolicyRule()
    policy_rule.description = 'block all principals from deleting projects, unless the principal is a member of project-admins@example.com and the project being deleted has a tag with the value test'
    policy_rule.deny_rule = deny_rule
    policy = types.Policy()
    policy.display_name = 'Restrict project deletion access'
    policy.rules = [policy_rule]
    request = types.CreatePolicyRequest()
    request.parent = f'policies/{attachment_point}/denypolicies'
    request.policy = policy
    request.policy_id = policy_id
    result = policies_client.create_policy(request=request).result()
    print(f"Created the deny policy: {result.name.rsplit('/')[-1]}")
if __name__ == '__main__':
    import uuid
    project_id = 'your-google-cloud-project-id'
    policy_id = f'deny-{uuid.uuid4()}'
    create_deny_policy(project_id, policy_id)