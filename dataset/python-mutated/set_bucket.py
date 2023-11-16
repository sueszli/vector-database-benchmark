from google.cloud import compute_v1

def set_usage_export_bucket(project_id: str, bucket_name: str, report_name_prefix: str='') -> None:
    if False:
        i = 10
        return i + 15
    '\n    Set Compute Engine usage export bucket for the Cloud project.\n    This sample presents how to interpret the default value for the\n    report name prefix parameter.\n\n    Args:\n        project_id: project ID or project number of the project to update.\n        bucket_name: Google Cloud Storage bucket used to store Compute Engine\n            usage reports. An existing Google Cloud Storage bucket is required.\n        report_name_prefix: Prefix of the usage report name which defaults to an empty string\n            to showcase default values behaviour.\n    '
    usage_export_location = compute_v1.UsageExportLocation()
    usage_export_location.bucket_name = bucket_name
    usage_export_location.report_name_prefix = report_name_prefix
    if not report_name_prefix:
        print('Setting report_name_prefix to empty value causes the report to have the default prefix of `usage_gce`.')
    projects_client = compute_v1.ProjectsClient()
    operation = projects_client.set_usage_export_bucket(project=project_id, usage_export_location_resource=usage_export_location)
    wait_for_extended_operation(operation, 'setting GCE usage bucket')