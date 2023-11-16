from google.cloud import dialogflowcx_v3

def sample_create_webhook():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3.WebhooksClient()
    webhook = dialogflowcx_v3.Webhook()
    webhook.generic_web_service.uri = 'uri_value'
    webhook.display_name = 'display_name_value'
    request = dialogflowcx_v3.CreateWebhookRequest(parent='parent_value', webhook=webhook)
    response = client.create_webhook(request=request)
    print(response)