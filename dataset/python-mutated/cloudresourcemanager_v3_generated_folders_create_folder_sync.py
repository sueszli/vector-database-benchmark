from google.cloud import resourcemanager_v3

def sample_create_folder():
    if False:
        print('Hello World!')
    client = resourcemanager_v3.FoldersClient()
    folder = resourcemanager_v3.Folder()
    folder.parent = 'parent_value'
    request = resourcemanager_v3.CreateFolderRequest(folder=folder)
    operation = client.create_folder(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)