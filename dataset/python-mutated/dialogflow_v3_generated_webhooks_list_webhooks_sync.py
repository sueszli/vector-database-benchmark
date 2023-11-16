from google.cloud import dialogflowcx_v3

def sample_list_webhooks():
    if False:
        return 10
    client = dialogflowcx_v3.WebhooksClient()
    request = dialogflowcx_v3.ListWebhooksRequest(parent='parent_value')
    page_result = client.list_webhooks(request=request)
    for response in page_result:
        print(response)