from google.cloud import bigquery
from google.cloud.storage import Client as storage_Client

def gcloud_bigquery_factory(context, request):
    if False:
        while True:
            i = 10
    service_account_info = request.registry.settings['gcloud.service_account_info']
    project = request.registry.settings['gcloud.project']
    return bigquery.Client.from_service_account_info(service_account_info, project=project)

def gcloud_gcs_factory(context, request):
    if False:
        for i in range(10):
            print('nop')
    service_account_info = request.registry.settings['gcloud.service_account_info']
    project = request.registry.settings['gcloud.project']
    return storage_Client.from_service_account_info(service_account_info, project=project)

def includeme(config):
    if False:
        return 10
    config.register_service_factory(gcloud_bigquery_factory, name='gcloud.bigquery')
    config.register_service_factory(gcloud_gcs_factory, name='gcloud.gcs')