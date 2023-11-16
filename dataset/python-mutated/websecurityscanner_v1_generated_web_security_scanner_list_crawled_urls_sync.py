from google.cloud import websecurityscanner_v1

def sample_list_crawled_urls():
    if False:
        i = 10
        return i + 15
    client = websecurityscanner_v1.WebSecurityScannerClient()
    request = websecurityscanner_v1.ListCrawledUrlsRequest()
    page_result = client.list_crawled_urls(request=request)
    for response in page_result:
        print(response)