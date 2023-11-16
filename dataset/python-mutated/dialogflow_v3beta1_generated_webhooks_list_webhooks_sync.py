from google.cloud import dialogflowcx_v3beta1

def sample_list_webhooks():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.WebhooksClient()
    request = dialogflowcx_v3beta1.ListWebhooksRequest(parent='parent_value')
    page_result = client.list_webhooks(request=request)
    for response in page_result:
        print(response)