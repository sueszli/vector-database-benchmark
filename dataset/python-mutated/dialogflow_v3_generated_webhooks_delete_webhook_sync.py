from google.cloud import dialogflowcx_v3

def sample_delete_webhook():
    if False:
        return 10
    client = dialogflowcx_v3.WebhooksClient()
    request = dialogflowcx_v3.DeleteWebhookRequest(name='name_value')
    client.delete_webhook(request=request)