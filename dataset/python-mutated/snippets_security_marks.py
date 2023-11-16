"""Demos for working with security marks."""

def add_to_asset(asset_name):
    if False:
        print('Hello World!')
    'Add new security marks to an asset.'
    from google.cloud import securitycenter
    from google.protobuf import field_mask_pb2
    client = securitycenter.SecurityCenterClient()
    marks_name = f'{asset_name}/securityMarks'
    field_mask = field_mask_pb2.FieldMask(paths=['marks.key_a', 'marks.key_b'])
    marks = {'key_a': 'value_a', 'key_b': 'value_b'}
    updated_marks = client.update_security_marks(request={'security_marks': {'name': marks_name, 'marks': marks}, 'update_mask': field_mask})
    print(updated_marks)
    return (updated_marks, marks)

def clear_from_asset(asset_name):
    if False:
        return 10
    'Removes security marks from an asset.'
    add_to_asset(asset_name)
    from google.cloud import securitycenter
    from google.protobuf import field_mask_pb2
    client = securitycenter.SecurityCenterClient()
    marks_name = f'{asset_name}/securityMarks'
    field_mask = field_mask_pb2.FieldMask(paths=['marks.key_a', 'marks.key_b'])
    updated_marks = client.update_security_marks(request={'security_marks': {'name': marks_name}, 'update_mask': field_mask})
    print(updated_marks)
    return updated_marks

def delete_and_update_marks(asset_name):
    if False:
        while True:
            i = 10
    'Updates and deletes security marks from an asset in the same call.'
    add_to_asset(asset_name)
    from google.cloud import securitycenter
    from google.protobuf import field_mask_pb2
    client = securitycenter.SecurityCenterClient()
    marks_name = f'{asset_name}/securityMarks'
    field_mask = field_mask_pb2.FieldMask(paths=['marks.key_a', 'marks.key_b'])
    marks = {'key_a': 'new_value_for_a'}
    updated_marks = client.update_security_marks(request={'security_marks': {'name': marks_name, 'marks': marks}, 'update_mask': field_mask})
    print(updated_marks)
    return updated_marks

def add_to_finding(finding_name):
    if False:
        for i in range(10):
            print('nop')
    'Adds security marks to a finding.'
    from google.cloud import securitycenter
    from google.protobuf import field_mask_pb2
    client = securitycenter.SecurityCenterClient()
    finding_marks_name = f'{finding_name}/securityMarks'
    field_mask = field_mask_pb2.FieldMask(paths=['marks.finding_key_a', 'marks.finding_key_b'])
    marks = {'finding_key_a': 'value_a', 'finding_key_b': 'value_b'}
    updated_marks = client.update_security_marks(request={'security_marks': {'name': finding_marks_name, 'marks': marks}, 'update_mask': field_mask})
    return (updated_marks, marks)

def list_assets_with_query_marks(organization_id, asset_name):
    if False:
        for i in range(10):
            print('nop')
    'Lists assets with a filter on security marks.'
    add_to_asset(asset_name)
    i = -1
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    parent = f'organizations/{organization_id}'
    marks_filter = 'security_marks.marks.key_a = "value_a"'
    asset_iterator = client.list_assets(request={'parent': parent, 'filter': marks_filter})
    for (i, asset_result) in enumerate(asset_iterator):
        print(i, asset_result)
    return i

def list_findings_with_query_marks(source_name, finding_name):
    if False:
        print('Hello World!')
    'Lists findings with a filter on security marks.'
    add_to_finding(finding_name)
    i = -1
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    marks_filter = 'NOT security_marks.marks.finding_key_a="value_a"'
    finding_iterator = client.list_findings(request={'parent': source_name, 'filter': marks_filter})
    for (i, finding_result) in enumerate(finding_iterator):
        print(i, finding_result)
    return i