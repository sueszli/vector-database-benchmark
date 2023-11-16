from google.cloud import videointelligence_v1beta2

def sample_annotate_video():
    if False:
        for i in range(10):
            print('nop')
    client = videointelligence_v1beta2.VideoIntelligenceServiceClient()
    request = videointelligence_v1beta2.AnnotateVideoRequest(features=['FACE_DETECTION'])
    operation = client.annotate_video(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)