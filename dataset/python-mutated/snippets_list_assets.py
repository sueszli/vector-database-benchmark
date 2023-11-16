""" Examples of listing assets in Security Command Center."""

def list_all_assets(organization_id):
    if False:
        return 10
    'Demonstrate listing and printing all assets.'
    i = 0
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    parent = f'organizations/{organization_id}'
    asset_iterator = client.list_assets(request={'parent': parent})
    for (i, asset_result) in enumerate(asset_iterator):
        print(i, asset_result)
    return i

def list_assets_with_filters(organization_id):
    if False:
        for i in range(10):
            print('nop')
    'Demonstrate listing assets with a filter.'
    i = 0
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    parent = f'organizations/{organization_id}'
    project_filter = 'security_center_properties.resource_type=' + '"google.cloud.resourcemanager.Project"'
    asset_iterator = client.list_assets(request={'parent': parent, 'filter': project_filter})
    for (i, asset_result) in enumerate(asset_iterator):
        print(i, asset_result)
    return i

def list_assets_with_filters_and_read_time(organization_id):
    if False:
        print('Hello World!')
    'Demonstrate listing assets with a filter.'
    i = 0
    from datetime import datetime, timedelta, timezone
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    parent = f'organizations/{organization_id}'
    project_filter = 'security_center_properties.resource_type=' + '"google.cloud.resourcemanager.Project"'
    read_time = datetime.now(tz=timezone.utc) - timedelta(days=1)
    asset_iterator = client.list_assets(request={'parent': parent, 'filter': project_filter, 'read_time': read_time})
    for (i, asset_result) in enumerate(asset_iterator):
        print(i, asset_result)
    return i

def list_point_in_time_changes(organization_id):
    if False:
        print('Hello World!')
    'Demonstrate listing assets along with their state changes.'
    i = 0
    from datetime import timedelta
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    parent = f'organizations/{organization_id}'
    project_filter = 'security_center_properties.resource_type=' + '"google.cloud.resourcemanager.Project"'
    compare_delta = timedelta(days=30)
    asset_iterator = client.list_assets(request={'parent': parent, 'filter': project_filter, 'compare_duration': compare_delta})
    for (i, asset) in enumerate(asset_iterator):
        print(i, asset)
    return i

def group_assets(organization_id):
    if False:
        while True:
            i = 10
    'Demonstrates grouping all assets by type.'
    i = 0
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    parent = f'organizations/{organization_id}'
    group_by_type = 'security_center_properties.resource_type'
    result_iterator = client.group_assets(request={'parent': parent, 'group_by': group_by_type})
    for (i, result) in enumerate(result_iterator):
        print(i + 1, result)
    return i

def group_filtered_assets(organization_id):
    if False:
        for i in range(10):
            print('nop')
    'Demonstrates grouping assets by type with a filter.'
    i = 0
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    org_name = f'organizations/{organization_id}'
    group_by_type = 'security_center_properties.resource_type'
    only_projects = 'security_center_properties.resource_type=' + '"google.cloud.resourcemanager.Project"'
    result_iterator = client.group_assets(request={'parent': org_name, 'group_by': group_by_type, 'filter': only_projects})
    for (i, result) in enumerate(result_iterator):
        print(i + 1, result)
    return i

def group_assets_by_changes(organization_id):
    if False:
        for i in range(10):
            print('nop')
    'Demonstrates grouping assets by their changes over a period of time.'
    i = 0
    from datetime import timedelta
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    duration = timedelta(days=5)
    parent = f'organizations/{organization_id}'
    result_iterator = client.group_assets(request={'parent': parent, 'group_by': 'state_change', 'compare_duration': duration})
    for (i, result) in enumerate(result_iterator):
        print(i + 1, result)
    return i