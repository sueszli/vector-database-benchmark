from google.cloud import resourcemanager_v3

def sample_undelete_folder():
    if False:
        return 10
    client = resourcemanager_v3.FoldersClient()
    request = resourcemanager_v3.UndeleteFolderRequest(name='name_value')
    operation = client.undelete_folder(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)