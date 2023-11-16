from google.cloud import resourcemanager_v3

def sample_list_folders():
    if False:
        while True:
            i = 10
    client = resourcemanager_v3.FoldersClient()
    request = resourcemanager_v3.ListFoldersRequest(parent='parent_value')
    page_result = client.list_folders(request=request)
    for response in page_result:
        print(response)