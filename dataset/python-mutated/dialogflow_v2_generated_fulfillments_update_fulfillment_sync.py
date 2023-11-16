from google.cloud import dialogflow_v2

def sample_update_fulfillment():
    if False:
        return 10
    client = dialogflow_v2.FulfillmentsClient()
    fulfillment = dialogflow_v2.Fulfillment()
    fulfillment.generic_web_service.uri = 'uri_value'
    fulfillment.name = 'name_value'
    request = dialogflow_v2.UpdateFulfillmentRequest(fulfillment=fulfillment)
    response = client.update_fulfillment(request=request)
    print(response)