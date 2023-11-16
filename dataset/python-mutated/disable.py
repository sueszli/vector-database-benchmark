from google.cloud import compute_v1

def disable_usage_export(project_id: str) -> None:
    if False:
        print('Hello World!')
    '\n    Disable Compute Engine usage export bucket for the Cloud Project.\n\n    Args:\n        project_id: project ID or project number of the project to update.\n    '
    projects_client = compute_v1.ProjectsClient()
    operation = projects_client.set_usage_export_bucket(project=project_id, usage_export_location_resource={})
    wait_for_extended_operation(operation, 'disabling GCE usage bucket')