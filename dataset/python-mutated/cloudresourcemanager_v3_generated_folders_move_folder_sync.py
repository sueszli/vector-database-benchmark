from google.cloud import resourcemanager_v3

def sample_move_folder():
    if False:
        print('Hello World!')
    client = resourcemanager_v3.FoldersClient()
    request = resourcemanager_v3.MoveFolderRequest(name='name_value', destination_parent='destination_parent_value')
    operation = client.move_folder(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)