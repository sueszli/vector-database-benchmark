from google.cloud import bigquery_biglake_v1alpha1

def sample_check_lock():
    if False:
        i = 10
        return i + 15
    client = bigquery_biglake_v1alpha1.MetastoreServiceClient()
    request = bigquery_biglake_v1alpha1.CheckLockRequest(name='name_value')
    response = client.check_lock(request=request)
    print(response)