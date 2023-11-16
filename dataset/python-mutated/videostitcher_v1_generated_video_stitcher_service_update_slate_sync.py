from google.cloud.video import stitcher_v1

def sample_update_slate():
    if False:
        while True:
            i = 10
    client = stitcher_v1.VideoStitcherServiceClient()
    request = stitcher_v1.UpdateSlateRequest()
    operation = client.update_slate(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)