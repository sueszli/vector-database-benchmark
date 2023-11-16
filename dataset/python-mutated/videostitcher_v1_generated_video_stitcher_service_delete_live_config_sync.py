from google.cloud.video import stitcher_v1

def sample_delete_live_config():
    if False:
        for i in range(10):
            print('nop')
    client = stitcher_v1.VideoStitcherServiceClient()
    request = stitcher_v1.DeleteLiveConfigRequest(name='name_value')
    operation = client.delete_live_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)