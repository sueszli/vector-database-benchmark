from google.cloud import dialogflowcx_v3

def sample_update_webhook():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3.WebhooksClient()
    webhook = dialogflowcx_v3.Webhook()
    webhook.generic_web_service.uri = 'uri_value'
    webhook.display_name = 'display_name_value'
    request = dialogflowcx_v3.UpdateWebhookRequest(webhook=webhook)
    response = client.update_webhook(request=request)
    print(response)