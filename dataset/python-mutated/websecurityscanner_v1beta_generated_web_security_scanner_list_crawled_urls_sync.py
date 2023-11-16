from google.cloud import websecurityscanner_v1beta

def sample_list_crawled_urls():
    if False:
        return 10
    client = websecurityscanner_v1beta.WebSecurityScannerClient()
    request = websecurityscanner_v1beta.ListCrawledUrlsRequest(parent='parent_value')
    page_result = client.list_crawled_urls(request=request)
    for response in page_result:
        print(response)