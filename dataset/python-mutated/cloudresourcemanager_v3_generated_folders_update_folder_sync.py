from google.cloud import resourcemanager_v3

def sample_update_folder():
    if False:
        i = 10
        return i + 15
    client = resourcemanager_v3.FoldersClient()
    folder = resourcemanager_v3.Folder()
    folder.parent = 'parent_value'
    request = resourcemanager_v3.UpdateFolderRequest(folder=folder)
    operation = client.update_folder(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)