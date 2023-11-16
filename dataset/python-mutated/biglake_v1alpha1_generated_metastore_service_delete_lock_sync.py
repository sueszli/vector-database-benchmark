from google.cloud import bigquery_biglake_v1alpha1

def sample_delete_lock():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_biglake_v1alpha1.MetastoreServiceClient()
    request = bigquery_biglake_v1alpha1.DeleteLockRequest(name='name_value')
    client.delete_lock(request=request)