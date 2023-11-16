from google.cloud import vmwareengine_v1

def get_vcenter_credentials(project_id: str, zone: str, private_cloud_name: str) -> vmwareengine_v1.Credentials:
    if False:
        print('Hello World!')
    '\n    Retrieves VCenter credentials for a Private Cloud.\n\n    Args:\n        project_id: name of the project hosting the private cloud.\n        zone: name of the zone hosting the private cloud.\n        private_cloud_name: name of the private cloud.\n\n    Returns:\n        A Credentials object.\n    '
    client = vmwareengine_v1.VmwareEngineClient()
    credentials = client.show_vcenter_credentials(private_cloud=f'projects/{project_id}/locations/{zone}/privateClouds/{private_cloud_name}')
    return credentials