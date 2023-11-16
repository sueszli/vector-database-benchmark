from google.cloud import bigquery_data_exchange_v1beta1

def sample_subscribe_listing():
    if False:
        return 10
    client = bigquery_data_exchange_v1beta1.AnalyticsHubServiceClient()
    destination_dataset = bigquery_data_exchange_v1beta1.DestinationDataset()
    destination_dataset.dataset_reference.dataset_id = 'dataset_id_value'
    destination_dataset.dataset_reference.project_id = 'project_id_value'
    destination_dataset.location = 'location_value'
    request = bigquery_data_exchange_v1beta1.SubscribeListingRequest(destination_dataset=destination_dataset, name='name_value')
    response = client.subscribe_listing(request=request)
    print(response)