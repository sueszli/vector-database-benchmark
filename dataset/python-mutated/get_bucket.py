from google.cloud import compute_v1

def get_usage_export_bucket(project_id: str) -> compute_v1.UsageExportLocation:
    if False:
        while True:
            i = 10
    '\n    Retrieve Compute Engine usage export bucket for the Cloud project.\n    Replaces the empty value returned by the API with the default value used\n    to generate report file names.\n\n    Args:\n        project_id: project ID or project number of the project to update.\n    Returns:\n        UsageExportLocation object describing the current usage export settings\n        for project project_id.\n    '
    projects_client = compute_v1.ProjectsClient()
    project_data = projects_client.get(project=project_id)
    uel = project_data.usage_export_location
    if not uel.bucket_name:
        return uel
    if not uel.report_name_prefix:
        print('Report name prefix not set, replacing with default value of `usage_gce`.')
        uel.report_name_prefix = 'usage_gce'
    return uel