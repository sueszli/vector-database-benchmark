from google.cloud import contact_center_insights_v1

def export_to_bigquery(project_id: str, bigquery_project_id: str, bigquery_dataset_id: str, bigquery_table_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Exports data to BigQuery.\n\n    Args:\n        project_id:\n            The project identifier that owns the data source to be exported.\n            For example, 'my-project'.\n        bigquery_project_id:\n            The project identifier that owns the BigQuery sink to export data to.\n            For example, 'my-project'.\n        bigquery_dataset_id:\n            The BigQuery dataset identifier. For example, 'my-dataset'.\n        bigquery_table_id:\n            The BigQuery table identifier. For example, 'my-table'.\n\n    Returns:\n        None.\n    "
    request = contact_center_insights_v1.ExportInsightsDataRequest()
    request.parent = contact_center_insights_v1.ContactCenterInsightsClient.common_location_path(project_id, 'us-central1')
    request.big_query_destination.project_id = bigquery_project_id
    request.big_query_destination.dataset = bigquery_dataset_id
    request.big_query_destination.table = bigquery_table_id
    request.filter = 'agent_id="007"'
    insights_client = contact_center_insights_v1.ContactCenterInsightsClient()
    export_operation = insights_client.export_insights_data(request=request)
    export_operation.result(timeout=600000)
    print('Exported data to BigQuery')