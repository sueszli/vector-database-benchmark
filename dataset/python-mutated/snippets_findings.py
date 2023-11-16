"""Examples of working with source and findings in Security Command Center."""

def create_source(organization_id):
    if False:
        print('Hello World!')
    'Create a new findings source.'
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    org_name = f'organizations/{organization_id}'
    created = client.create_source(request={'parent': org_name, 'source': {'display_name': 'Customized Display Name', 'description': 'A new custom source that does X'}})
    print(f'Created Source: {created.name}')

def get_source(source_name):
    if False:
        for i in range(10):
            print('nop')
    'Gets an existing source.'
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    source = client.get_source(request={'name': source_name})
    print(f'Source: {source}')
    return source

def update_source(source_name):
    if False:
        return 10
    "Updates a source's display name."
    from google.cloud import securitycenter
    from google.protobuf import field_mask_pb2
    client = securitycenter.SecurityCenterClient()
    field_mask = field_mask_pb2.FieldMask(paths=['display_name'])
    updated = client.update_source(request={'source': {'name': source_name, 'display_name': 'Updated Display Name'}, 'update_mask': field_mask})
    print(f'Updated Source: {updated}')
    return updated

def add_user_to_source(source_name):
    if False:
        i = 10
        return i + 15
    'Gives a user findingsEditor permission to the source.'
    user_email = 'csccclienttest@gmail.com'
    from google.cloud import securitycenter
    from google.iam.v1 import policy_pb2
    client = securitycenter.SecurityCenterClient()
    old_policy = client.get_iam_policy(request={'resource': source_name})
    print(f'Old Policy: {old_policy}')
    binding = policy_pb2.Binding()
    binding.role = 'roles/securitycenter.findingsEditor'
    binding.members.append(f'user:{user_email}')
    updated = client.set_iam_policy(request={'resource': source_name, 'policy': {'etag': old_policy.etag, 'bindings': [binding]}})
    print(f'Updated Policy: {updated}')
    return (binding, updated)

def list_source(organization_id):
    if False:
        while True:
            i = 10
    'Lists finding sources.'
    i = -1
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    parent = f'organizations/{organization_id}'
    for (i, source) in enumerate(client.list_sources(request={'parent': parent})):
        print(i, source)
    return i

def create_finding(source_name, finding_id):
    if False:
        i = 10
        return i + 15
    'Creates a new finding.'
    import datetime
    from google.cloud import securitycenter
    from google.cloud.securitycenter_v1 import Finding
    client = securitycenter.SecurityCenterClient()
    event_time = datetime.datetime.now(tz=datetime.timezone.utc)
    resource_name = '//cloudresourcemanager.googleapis.com/organizations/11232'
    finding = Finding(state=Finding.State.ACTIVE, resource_name=resource_name, category='MEDIUM_RISK_ONE', event_time=event_time)
    created_finding = client.create_finding(request={'parent': source_name, 'finding_id': finding_id, 'finding': finding})
    print(created_finding)
    return created_finding

def create_finding_with_source_properties(source_name):
    if False:
        while True:
            i = 10
    'Demonstrate creating a new finding with source properties.'
    import datetime
    from google.cloud import securitycenter
    from google.cloud.securitycenter_v1 import Finding
    from google.protobuf.struct_pb2 import Value
    client = securitycenter.SecurityCenterClient()
    finding_id = 'samplefindingid2'
    resource_name = '//cloudresourcemanager.googleapis.com/organizations/11232'
    str_value = Value()
    str_value.string_value = 'string_example'
    num_value = Value()
    num_value.number_value = 1234
    event_time = datetime.datetime.now(tz=datetime.timezone.utc)
    finding = Finding(state=Finding.State.ACTIVE, resource_name=resource_name, category='MEDIUM_RISK_ONE', source_properties={'s_value': 'string_example', 'n_value': 1234}, event_time=event_time)
    created_finding = client.create_finding(request={'parent': source_name, 'finding_id': finding_id, 'finding': finding})
    print(created_finding)

def update_finding(source_name):
    if False:
        return 10
    import datetime
    from google.cloud import securitycenter
    from google.cloud.securitycenter_v1 import Finding
    from google.protobuf import field_mask_pb2
    client = securitycenter.SecurityCenterClient()
    field_mask = field_mask_pb2.FieldMask(paths=['source_properties.s_value', 'event_time'])
    event_time = datetime.datetime.now(tz=datetime.timezone.utc)
    finding_name = f'{source_name}/findings/samplefindingid2'
    finding = Finding(name=finding_name, source_properties={'s_value': 'new_string'}, event_time=event_time)
    updated_finding = client.update_finding(request={'finding': finding, 'update_mask': field_mask})
    print('New Source properties: {}, Event Time {}'.format(updated_finding.source_properties, updated_finding.event_time))

def update_finding_state(source_name):
    if False:
        i = 10
        return i + 15
    'Demonstrate updating only a finding state.'
    import datetime
    from google.cloud import securitycenter
    from google.cloud.securitycenter_v1 import Finding
    client = securitycenter.SecurityCenterClient()
    finding_name = f'{source_name}/findings/samplefindingid2'
    new_finding = client.set_finding_state(request={'name': finding_name, 'state': Finding.State.INACTIVE, 'start_time': datetime.datetime.now(tz=datetime.timezone.utc)})
    print(f'New state: {new_finding.state}')

def trouble_shoot(source_name):
    if False:
        return 10
    'Demonstrate calling test_iam_permissions to determine if the\n    service account has the correct permisions.'
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    permission_response = client.test_iam_permissions(request={'resource': source_name, 'permissions': ['securitycenter.findings.update']})
    print('Permision to create or update findings? {}'.format(len(permission_response.permissions) > 0))
    assert len(permission_response.permissions) > 0
    permission_response = client.test_iam_permissions(request={'resource': source_name, 'permissions': ['securitycenter.findings.setState']})
    print(f'Permision to update state? {len(permission_response.permissions) > 0}')
    return permission_response
    assert len(permission_response.permissions) > 0

def list_all_findings(organization_id):
    if False:
        return 10
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    parent = f'organizations/{organization_id}'
    all_sources = f'{parent}/sources/-'
    finding_result_iterator = client.list_findings(request={'parent': all_sources})
    for (i, finding_result) in enumerate(finding_result_iterator):
        print('{}: name: {} resource: {}'.format(i, finding_result.finding.name, finding_result.finding.resource_name))
    return i

def list_filtered_findings(source_name):
    if False:
        i = 10
        return i + 15
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    finding_result_iterator = client.list_findings(request={'parent': source_name, 'filter': 'category="MEDIUM_RISK_ONE"'})
    for (i, finding_result) in enumerate(finding_result_iterator):
        print('{}: name: {} resource: {}'.format(i, finding_result.finding.name, finding_result.finding.resource_name))
    return i

def list_findings_at_time(source_name):
    if False:
        while True:
            i = 10
    from datetime import datetime, timedelta
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    five_days_ago = str(datetime.now() - timedelta(days=5))
    i = -1
    finding_result_iterator = client.list_findings(request={'parent': source_name, 'filter': five_days_ago})
    for (i, finding_result) in enumerate(finding_result_iterator):
        print('{}: name: {} resource: {}'.format(i, finding_result.finding.name, finding_result.finding.resource_name))
    return i

def get_iam_policy(source_name):
    if False:
        for i in range(10):
            print('nop')
    'Gives a user findingsEditor permission to the source.'
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    policy = client.get_iam_policy(request={'resource': source_name})
    print(f'Policy: {policy}')

def group_all_findings(organization_id):
    if False:
        for i in range(10):
            print('nop')
    'Demonstrates grouping all findings across an organization.'
    i = 0
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    parent = f'organizations/{organization_id}'
    all_sources = f'{parent}/sources/-'
    group_result_iterator = client.group_findings(request={'parent': all_sources, 'group_by': 'category'})
    for (i, group_result) in enumerate(group_result_iterator):
        print(i + 1, group_result)
    return i

def group_filtered_findings(source_name):
    if False:
        print('Hello World!')
    'Demonstrates grouping all findings across an organization.'
    i = 0
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    group_result_iterator = client.group_findings(request={'parent': source_name, 'group_by': 'category', 'filter': 'state="ACTIVE"'})
    for (i, group_result) in enumerate(group_result_iterator):
        print(i + 1, group_result)
    return i

def group_findings_at_time(source_name):
    if False:
        return 10
    'Demonstrates grouping all findings across an organization as of\n    a specific time.'
    i = -1
    from datetime import datetime, timedelta
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    read_time = datetime.utcnow() - timedelta(days=1)
    group_result_iterator = client.group_findings(request={'parent': source_name, 'group_by': 'category', 'read_time': read_time})
    for (i, group_result) in enumerate(group_result_iterator):
        print(i + 1, group_result)
    return i

def group_findings_and_changes(source_name):
    if False:
        i = 10
        return i + 15
    'Demonstrates grouping all findings across an organization and\n    associated changes.'
    i = 0
    from datetime import timedelta
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    compare_delta = timedelta(days=30)
    group_result_iterator = client.group_findings(request={'parent': source_name, 'group_by': 'state_change', 'compare_duration': compare_delta})
    for (i, group_result) in enumerate(group_result_iterator):
        print(i + 1, group_result)
    return i