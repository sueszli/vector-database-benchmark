from google.cloud import websecurityscanner_v1alpha

def sample_list_crawled_urls():
    if False:
        for i in range(10):
            print('nop')
    client = websecurityscanner_v1alpha.WebSecurityScannerClient()
    request = websecurityscanner_v1alpha.ListCrawledUrlsRequest(parent='parent_value')
    page_result = client.list_crawled_urls(request=request)
    for response in page_result:
        print(response)