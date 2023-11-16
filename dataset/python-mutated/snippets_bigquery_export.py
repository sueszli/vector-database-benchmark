"""Snippets on exporting findings from Security Command Center to BigQuery."""

def create_bigquery_export(parent: str, export_filter: str, bigquery_dataset_id: str, bigquery_export_id: str):
    if False:
        while True:
            i = 10
    from google.cloud import securitycenter
    '\n    Create export configuration to export findings from a project to a BigQuery dataset.\n    Optionally specify filter to export certain findings only.\n\n    Args:\n        parent: Use any one of the following resource paths:\n             - organizations/{organization_id}\n             - folders/{folder_id}\n             - projects/{project_id}\n        export_filter: Expression that defines the filter to apply across create/update events of findings.\n        bigquery_dataset_id: The BigQuery dataset to write findings\' updates to.\n        bigquery_export_id: Unique identifier provided by the client.\n             - example id: f"default-{str(uuid.uuid4()).split(\'-\')[0]}"\n        For more info, see:\n        https://cloud.google.com/security-command-center/docs/how-to-analyze-findings-in-big-query#export_findings_from_to\n    '
    client = securitycenter.SecurityCenterClient()
    bigquery_export = securitycenter.BigQueryExport()
    bigquery_export.description = 'Export low and medium findings if the compute resource has an IAM anomalous grant'
    bigquery_export.filter = export_filter
    bigquery_export.dataset = f'{parent}/datasets/{bigquery_dataset_id}'
    request = securitycenter.CreateBigQueryExportRequest()
    request.parent = parent
    request.big_query_export = bigquery_export
    request.big_query_export_id = bigquery_export_id
    response = client.create_big_query_export(request)
    print(f'BigQuery export request created successfully: {response.name}\n')

def get_bigquery_export(parent: str, bigquery_export_id: str):
    if False:
        return 10
    from google.cloud import securitycenter
    '\n    Retrieve an existing BigQuery export.\n    Args:\n        parent: Use any one of the following resource paths:\n                 - organizations/{organization_id}\n                 - folders/{folder_id}\n                 - projects/{project_id}\n        bigquery_export_id: Unique identifier that is used to identify the export.\n    '
    client = securitycenter.SecurityCenterClient()
    request = securitycenter.GetBigQueryExportRequest()
    request.name = f'{parent}/bigQueryExports/{bigquery_export_id}'
    response = client.get_big_query_export(request)
    print(f'Retrieved the BigQuery export: {response.name}')

def list_bigquery_exports(parent: str):
    if False:
        while True:
            i = 10
    from google.cloud import securitycenter
    '\n    List BigQuery exports in the given parent.\n    Args:\n         parent: The parent which owns the collection of BigQuery exports.\n             Use any one of the following resource paths:\n                 - organizations/{organization_id}\n                 - folders/{folder_id}\n                 - projects/{project_id}\n    '
    client = securitycenter.SecurityCenterClient()
    request = securitycenter.ListBigQueryExportsRequest()
    request.parent = parent
    response = client.list_big_query_exports(request)
    print('Listing BigQuery exports:')
    for bigquery_export in response:
        print(bigquery_export.name)

def update_bigquery_export(parent: str, export_filter: str, bigquery_export_id: str):
    if False:
        return 10
    '\n    Updates an existing BigQuery export.\n    Args:\n        parent: Use any one of the following resource paths:\n                 - organizations/{organization_id}\n                 - folders/{folder_id}\n                 - projects/{project_id}\n        export_filter: Expression that defines the filter to apply across create/update events of findings.\n        bigquery_export_id: Unique identifier provided by the client.\n        For more info, see:\n        https://cloud.google.com/security-command-center/docs/how-to-analyze-findings-in-big-query#export_findings_from_to\n    '
    from google.cloud import securitycenter
    from google.protobuf import field_mask_pb2
    client = securitycenter.SecurityCenterClient()
    bigquery_export = securitycenter.BigQueryExport()
    bigquery_export.name = f'{parent}/bigQueryExports/{bigquery_export_id}'
    bigquery_export.filter = export_filter
    field_mask = field_mask_pb2.FieldMask(paths=['filter'])
    request = securitycenter.UpdateBigQueryExportRequest()
    request.big_query_export = bigquery_export
    request.update_mask = field_mask
    response = client.update_big_query_export(request)
    if response.filter != export_filter:
        print('Failed to update BigQueryExport!')
        return
    print('BigQueryExport updated successfully!')

def delete_bigquery_export(parent: str, bigquery_export_id: str):
    if False:
        print('Hello World!')
    '\n    Delete an existing BigQuery export.\n    Args:\n        parent: Use any one of the following resource paths:\n                 - organizations/{organization_id}\n                 - folders/{folder_id}\n                 - projects/{project_id}\n        bigquery_export_id: Unique identifier that is used to identify the export.\n    '
    from google.cloud import securitycenter
    client = securitycenter.SecurityCenterClient()
    request = securitycenter.DeleteBigQueryExportRequest()
    request.name = f'{parent}/bigQueryExports/{bigquery_export_id}'
    client.delete_big_query_export(request)
    print(f'BigQuery export request deleted successfully: {bigquery_export_id}')