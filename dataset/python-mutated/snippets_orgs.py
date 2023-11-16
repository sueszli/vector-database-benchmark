"""Examples for working with organization settings. """

def get_settings(organization_id):
    if False:
        i = 10
        return i + 15
    'Example showing how to retreive current organization settings.'
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    org_settings_name = client.organization_settings_path(organization_id)
    org_settings = client.get_organization_settings(request={'name': org_settings_name})
    print(org_settings)

def update_asset_discovery_org_settings(organization_id):
    if False:
        for i in range(10):
            print('nop')
    'Example showing how to update the asset discovery configuration\n    for an organization.'
    from google.cloud import securitycenter
    from google.protobuf import field_mask_pb2
    client = securitycenter.SecurityCenterClient()
    org_settings_name = 'organizations/{org_id}/organizationSettings'.format(org_id=organization_id)
    field_mask = field_mask_pb2.FieldMask(paths=['enable_asset_discovery'])
    updated = client.update_organization_settings(request={'organization_settings': {'name': org_settings_name, 'enable_asset_discovery': True}, 'update_mask': field_mask})
    print(f'Asset Discovery Enabled? {updated.enable_asset_discovery}')
    return updated