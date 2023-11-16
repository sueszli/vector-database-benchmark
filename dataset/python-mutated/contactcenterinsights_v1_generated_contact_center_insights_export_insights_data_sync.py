from google.cloud import contact_center_insights_v1

def sample_export_insights_data():
    if False:
        i = 10
        return i + 15
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    big_query_destination = contact_center_insights_v1.BigQueryDestination()
    big_query_destination.dataset = 'dataset_value'
    request = contact_center_insights_v1.ExportInsightsDataRequest(big_query_destination=big_query_destination, parent='parent_value')
    operation = client.export_insights_data(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)