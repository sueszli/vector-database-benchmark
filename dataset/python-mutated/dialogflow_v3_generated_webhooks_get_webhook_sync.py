from google.cloud import dialogflowcx_v3

def sample_get_webhook():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3.WebhooksClient()
    request = dialogflowcx_v3.GetWebhookRequest(name='name_value')
    response = client.get_webhook(request=request)
    print(response)