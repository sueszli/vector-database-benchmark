from google.cloud import dialogflowcx_v3beta1

def sample_get_webhook():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3beta1.WebhooksClient()
    request = dialogflowcx_v3beta1.GetWebhookRequest(name='name_value')
    response = client.get_webhook(request=request)
    print(response)