def lookup_entry(override_values):
    if False:
        while True:
            i = 10
    'Retrieves Data Catalog entry for the given Google Cloud Platform resource.'
    from google.cloud import datacatalog_v1
    datacatalog = datacatalog_v1.DataCatalogClient()
    bigquery_project_id = 'my_bigquery_project'
    dataset_id = 'my_dataset'
    table_id = 'my_table'
    pubsub_project_id = 'my_pubsub_project'
    topic_id = 'my_topic'
    bigquery_project_id = override_values.get('bigquery_project_id', bigquery_project_id)
    dataset_id = override_values.get('dataset_id', dataset_id)
    table_id = override_values.get('table_id', table_id)
    pubsub_project_id = override_values.get('pubsub_project_id', pubsub_project_id)
    topic_id = override_values.get('topic_id', topic_id)
    resource_name = f'//bigquery.googleapis.com/projects/{bigquery_project_id}/datasets/{dataset_id}'
    entry = datacatalog.lookup_entry(request={'linked_resource': resource_name})
    print(f'Retrieved entry {entry.name} for BigQuery Dataset resource {entry.linked_resource}')
    sql_resource = f'bigquery.dataset.`{bigquery_project_id}`.`{dataset_id}`'
    entry = datacatalog.lookup_entry(request={'sql_resource': sql_resource})
    print(f'Retrieved entry {entry.name} for BigQuery Dataset resource {entry.linked_resource}')
    resource_name = f'//bigquery.googleapis.com/projects/{bigquery_project_id}/datasets/{dataset_id}/tables/{table_id}'
    entry = datacatalog.lookup_entry(request={'linked_resource': resource_name})
    print(f'Retrieved entry {entry.name} for BigQuery Table {entry.linked_resource}')
    sql_resource = f'bigquery.table.`{bigquery_project_id}`.`{dataset_id}`.`{table_id}`'
    entry = datacatalog.lookup_entry(request={'sql_resource': sql_resource})
    print(f'Retrieved entry {entry.name} for BigQuery Table resource {entry.linked_resource}')
    resource_name = f'//pubsub.googleapis.com/projects/{pubsub_project_id}/topics/{topic_id}'
    entry = datacatalog.lookup_entry(request={'linked_resource': resource_name})
    print(f'Retrieved entry {entry.name} for Pub/Sub Topic resource {entry.linked_resource}')
    sql_resource = f'pubsub.topic.`{pubsub_project_id}`.`{topic_id}`'
    entry = datacatalog.lookup_entry(request={'sql_resource': sql_resource})
    print(f'Retrieved entry {entry.name} for Pub/Sub Topic resource {entry.linked_resource}')