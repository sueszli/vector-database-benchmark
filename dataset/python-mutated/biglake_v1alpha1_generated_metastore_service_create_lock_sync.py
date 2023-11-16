from google.cloud import bigquery_biglake_v1alpha1

def sample_create_lock():
    if False:
        print('Hello World!')
    client = bigquery_biglake_v1alpha1.MetastoreServiceClient()
    lock = bigquery_biglake_v1alpha1.Lock()
    lock.table_id = 'table_id_value'
    request = bigquery_biglake_v1alpha1.CreateLockRequest(parent='parent_value', lock=lock)
    response = client.create_lock(request=request)
    print(response)