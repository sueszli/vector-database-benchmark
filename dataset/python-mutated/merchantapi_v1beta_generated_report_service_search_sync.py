from google.shopping import merchant_reports_v1beta

def sample_search():
    if False:
        for i in range(10):
            print('nop')
    client = merchant_reports_v1beta.ReportServiceClient()
    request = merchant_reports_v1beta.SearchRequest(parent='parent_value', query='query_value')
    page_result = client.search(request=request)
    for response in page_result:
        print(response)