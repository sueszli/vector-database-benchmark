from google.cloud import resourcemanager_v3

def sample_delete_folder():
    if False:
        while True:
            i = 10
    client = resourcemanager_v3.FoldersClient()
    request = resourcemanager_v3.DeleteFolderRequest(name='name_value')
    operation = client.delete_folder(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)