def create_mute_rule(parent_path: str, mute_config_id: str) -> None:
    if False:
        print('Hello World!')
    '\n    Creates a mute configuration under a given scope that will mute\n    all new findings that match a given filter.\n    Existing findings will NOT BE muted.\n    Args:\n        parent_path: use any one of the following options:\n                     - organizations/{organization_id}\n                     - folders/{folder_id}\n                     - projects/{project_id}\n        mute_config_id: Set a unique id; max of 63 chars.\n    '
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    mute_config = securitycenter.MuteConfig()
    mute_config.description = "Mute low-medium IAM grants excluding 'compute' "
    mute_config.filter = 'severity="LOW" OR severity="MEDIUM" AND category="Persistence: IAM Anomalous Grant" AND -resource.type:"compute"'
    request = securitycenter.CreateMuteConfigRequest()
    request.parent = parent_path
    request.mute_config_id = mute_config_id
    request.mute_config = mute_config
    mute_config = client.create_mute_config(request=request)
    print(f'Mute rule created successfully: {mute_config.name}')

def delete_mute_rule(mute_config_name: str) -> None:
    if False:
        while True:
            i = 10
    '\n    Deletes a mute configuration given its resource name.\n    Note: Previously muted findings are not affected when a mute config is deleted.\n    Args:\n        mute_config_name: Specify the name of the mute config to delete.\n                          Use any one of the following formats:\n                          - organizations/{organization}/muteConfigs/{config_id}\n                          - folders/{folder}/muteConfigs/{config_id} or\n                          - projects/{project}/muteConfigs/{config_id}\n    '
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    request = securitycenter.DeleteMuteConfigRequest()
    request.name = mute_config_name
    client.delete_mute_config(request)
    print(f'Mute rule deleted successfully: {mute_config_name}')

def get_mute_rule(mute_config_name: str) -> None:
    if False:
        while True:
            i = 10
    '\n    Retrieves a mute configuration given its resource name.\n    Args:\n        mute_config_name: Name of the mute config to retrieve.\n                          Use any one of the following formats:\n                          - organizations/{organization}/muteConfigs/{config_id}\n                          - folders/{folder}/muteConfigs/{config_id}\n                          - projects/{project}/muteConfigs/{config_id}\n    '
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    request = securitycenter.GetMuteConfigRequest()
    request.name = mute_config_name
    mute_config = client.get_mute_config(request)
    print(f'Retrieved the mute rule: {mute_config.name}')

def list_mute_rules(parent: str) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Listing mute configs at organization level will return all the configs\n    at the org, folder and project levels.\n    Similarly, listing configs at folder level will list all the configs\n    at the folder and project levels.\n    Args:\n        parent: Use any one of the following resource paths to list mute configurations:\n                - organizations/{organization_id}\n                - folders/{folder_id}\n                - projects/{project_id}\n    '
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    request = securitycenter.ListMuteConfigsRequest()
    request.parent = parent
    for mute_config in client.list_mute_configs(request):
        print(mute_config.name)

def update_mute_rule(mute_config_name: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Updates an existing mute configuration.\n    The following can be updated in a mute config: description, and filter/ mute rule.\n    Args:\n        mute_config_name: Specify the name of the mute config to delete.\n                          Use any one of the following formats:\n                          - organizations/{organization}/muteConfigs/{config_id}\n                          - folders/{folder}/muteConfigs/{config_id}\n                          - projects/{project}/muteConfigs/{config_id}\n    '
    from google.cloud import securitycenter
    from google.protobuf import field_mask_pb2
    client = securitycenter.SecurityCenterClient()
    update_mute_config = securitycenter.MuteConfig()
    update_mute_config.name = mute_config_name
    update_mute_config.description = 'Updated mute config description'
    field_mask = field_mask_pb2.FieldMask(paths=['description'])
    request = securitycenter.UpdateMuteConfigRequest()
    request.mute_config = update_mute_config
    request.update_mask = field_mask
    mute_config = client.update_mute_config(request)
    print(f'Updated mute rule : {mute_config}')

def set_mute_finding(finding_path: str) -> None:
    if False:
        return 10
    '\n      Mute an individual finding.\n      If a finding is already muted, muting it again has no effect.\n      Various mute states are: MUTE_UNSPECIFIED/MUTE/UNMUTE.\n    Args:\n        finding_path: The relative resource name of the finding. See:\n        https://cloud.google.com/apis/design/resource_names#relative_resource_name\n        Use any one of the following formats:\n        - organizations/{organization_id}/sources/{source_id}/finding/{finding_id},\n        - folders/{folder_id}/sources/{source_id}/finding/{finding_id},\n        - projects/{project_id}/sources/{source_id}/finding/{finding_id}.\n    '
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    request = securitycenter.SetMuteRequest()
    request.name = finding_path
    request.mute = securitycenter.Finding.Mute.MUTED
    finding = client.set_mute(request)
    print(f'Mute value for the finding: {finding.mute.name}')

def set_unmute_finding(finding_path: str) -> None:
    if False:
        print('Hello World!')
    "\n      Unmute an individual finding.\n      Unmuting a finding that isn't muted has no effect.\n      Various mute states are: MUTE_UNSPECIFIED/MUTE/UNMUTE.\n    Args:\n        finding_path: The relative resource name of the finding. See:\n        https://cloud.google.com/apis/design/resource_names#relative_resource_name\n        Use any one of the following formats:\n        - organizations/{organization_id}/sources/{source_id}/finding/{finding_id},\n        - folders/{folder_id}/sources/{source_id}/finding/{finding_id},\n        - projects/{project_id}/sources/{source_id}/finding/{finding_id}.\n    "
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    request = securitycenter.SetMuteRequest()
    request.name = finding_path
    request.mute = securitycenter.Finding.Mute.UNMUTED
    finding = client.set_mute(request)
    print(f'Mute value for the finding: {finding.mute.name}')

def bulk_mute_findings(parent_path: str, mute_rule: str) -> None:
    if False:
        print('Hello World!')
    '\n      Kicks off a long-running operation (LRO) to bulk mute findings for a parent based on a filter.\n      The parent can be either an organization, folder, or project. The findings\n      matched by the filter will be muted after the LRO is done.\n    Args:\n        parent_path: use any one of the following options:\n                     - organizations/{organization}\n                     - folders/{folder}\n                     - projects/{project}\n        mute_rule: Expression that identifies findings that should be updated.\n    '
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    request = securitycenter.BulkMuteFindingsRequest()
    request.parent = parent_path
    request.filter = mute_rule
    response = client.bulk_mute_findings(request)
    print(f'Bulk mute findings completed successfully! : {response}')