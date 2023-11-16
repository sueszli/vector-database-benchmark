from google.cloud.video import stitcher_v1

def sample_create_slate():
    if False:
        for i in range(10):
            print('nop')
    client = stitcher_v1.VideoStitcherServiceClient()
    request = stitcher_v1.CreateSlateRequest(parent='parent_value', slate_id='slate_id_value')
    operation = client.create_slate(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)