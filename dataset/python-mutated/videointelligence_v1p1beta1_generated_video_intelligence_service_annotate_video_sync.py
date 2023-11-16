from google.cloud import videointelligence_v1p1beta1

def sample_annotate_video():
    if False:
        i = 10
        return i + 15
    client = videointelligence_v1p1beta1.VideoIntelligenceServiceClient()
    request = videointelligence_v1p1beta1.AnnotateVideoRequest(features=['SPEECH_TRANSCRIPTION'])
    operation = client.annotate_video(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)