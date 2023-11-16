from google.cloud import videointelligence_v1p3beta1

def sample_annotate_video():
    if False:
        return 10
    client = videointelligence_v1p3beta1.VideoIntelligenceServiceClient()
    request = videointelligence_v1p3beta1.AnnotateVideoRequest(features=['PERSON_DETECTION'])
    operation = client.annotate_video(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)