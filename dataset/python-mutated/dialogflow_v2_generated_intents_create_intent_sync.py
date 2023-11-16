from google.cloud import dialogflow_v2

def sample_create_intent():
    if False:
        print('Hello World!')
    client = dialogflow_v2.IntentsClient()
    intent = dialogflow_v2.Intent()
    intent.display_name = 'display_name_value'
    request = dialogflow_v2.CreateIntentRequest(parent='parent_value', intent=intent)
    response = client.create_intent(request=request)
    print(response)