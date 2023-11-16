from google.cloud import resourcemanager_v3

def sample_search_folders():
    if False:
        for i in range(10):
            print('nop')
    client = resourcemanager_v3.FoldersClient()
    request = resourcemanager_v3.SearchFoldersRequest()
    page_result = client.search_folders(request=request)
    for response in page_result:
        print(response)