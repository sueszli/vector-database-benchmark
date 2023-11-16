from google.cloud import resourcemanager_v3

def sample_get_folder():
    if False:
        for i in range(10):
            print('nop')
    client = resourcemanager_v3.FoldersClient()
    request = resourcemanager_v3.GetFolderRequest(name='name_value')
    response = client.get_folder(request=request)
    print(response)