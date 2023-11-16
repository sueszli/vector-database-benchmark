from google.cloud import videointelligence_v1p2beta1

def sample_annotate_video():
    if False:
        while True:
            i = 10
    client = videointelligence_v1p2beta1.VideoIntelligenceServiceClient()
    request = videointelligence_v1p2beta1.AnnotateVideoRequest(features=['OBJECT_TRACKING'])
    operation = client.annotate_video(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)