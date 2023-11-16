from google.cloud import bigquery_analyticshub_v1

def sample_subscribe_listing():
    if False:
        i = 10
        return i + 15
    client = bigquery_analyticshub_v1.AnalyticsHubServiceClient()
    destination_dataset = bigquery_analyticshub_v1.DestinationDataset()
    destination_dataset.dataset_reference.dataset_id = 'dataset_id_value'
    destination_dataset.dataset_reference.project_id = 'project_id_value'
    destination_dataset.location = 'location_value'
    request = bigquery_analyticshub_v1.SubscribeListingRequest(destination_dataset=destination_dataset, name='name_value')
    response = client.subscribe_listing(request=request)
    print(response)