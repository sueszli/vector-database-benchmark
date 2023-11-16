from google.cloud import osconfig_v1

def sample_create_os_policy_assignment():
    if False:
        for i in range(10):
            print('nop')
    client = osconfig_v1.OsConfigZonalServiceClient()
    os_policy_assignment = osconfig_v1.OSPolicyAssignment()
    os_policy_assignment.os_policies.id = 'id_value'
    os_policy_assignment.os_policies.mode = 'ENFORCEMENT'
    os_policy_assignment.os_policies.resource_groups.resources.pkg.apt.name = 'name_value'
    os_policy_assignment.os_policies.resource_groups.resources.pkg.desired_state = 'REMOVED'
    os_policy_assignment.os_policies.resource_groups.resources.id = 'id_value'
    os_policy_assignment.rollout.disruption_budget.fixed = 528
    request = osconfig_v1.CreateOSPolicyAssignmentRequest(parent='parent_value', os_policy_assignment=os_policy_assignment, os_policy_assignment_id='os_policy_assignment_id_value')
    operation = client.create_os_policy_assignment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)