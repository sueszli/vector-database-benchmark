from google.cloud.video import stitcher_v1

def sample_create_live_config():
    if False:
        print('Hello World!')
    client = stitcher_v1.VideoStitcherServiceClient()
    live_config = stitcher_v1.LiveConfig()
    live_config.source_uri = 'source_uri_value'
    live_config.ad_tracking = 'SERVER'
    request = stitcher_v1.CreateLiveConfigRequest(parent='parent_value', live_config_id='live_config_id_value', live_config=live_config)
    operation = client.create_live_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)