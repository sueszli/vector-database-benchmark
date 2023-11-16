"""Helper functions to generate resource labels strings for GCP entitites

These can be used on MonitoringInfo 'resource' labels.

See example entities:
    https://s.apache.org/beam-gcp-debuggability

For GCP entities, populate the RESOURCE label with the aip.dev/122 format:
https://google.aip.dev/122

If an official GCP format does not exist, try to use the following format.
    //whatever.googleapis.com/parents/{parentId}/whatevers/{whateverId}
"""

def BigQueryTable(project_id, dataset_id, table_id):
    if False:
        print('Hello World!')
    return '//bigquery.googleapis.com/projects/%s/datasets/%s/tables/%s' % (project_id, dataset_id, table_id)

def GoogleCloudStorageBucket(bucket_id):
    if False:
        i = 10
        return i + 15
    return '//storage.googleapis.com/buckets/%s' % bucket_id

def DatastoreNamespace(project_id, namespace_id):
    if False:
        print('Hello World!')
    return '//bigtable.googleapis.com/projects/%s/namespaces/%s' % (project_id, namespace_id)

def SpannerTable(project_id, database_id, table_id):
    if False:
        for i in range(10):
            print('nop')
    return '//spanner.googleapis.com/projects/%s/topics/%s/tables/%s' % (project_id, database_id, table_id)

def SpannerSqlQuery(project_id, query_name):
    if False:
        i = 10
        return i + 15
    return '//spanner.googleapis.com/projects/%s/queries/%s' % (project_id, query_name)

def BigtableTable(project_id, instance_id, table_id):
    if False:
        print('Hello World!')
    return '//bigtable.googleapis.com/projects/%s/instances/%s/tables/%s' % (project_id, instance_id, table_id)