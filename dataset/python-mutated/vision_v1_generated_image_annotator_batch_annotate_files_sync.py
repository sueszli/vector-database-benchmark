from google.cloud import vision_v1

def sample_batch_annotate_files():
    if False:
        while True:
            i = 10
    client = vision_v1.ImageAnnotatorClient()
    request = vision_v1.BatchAnnotateFilesRequest()
    response = client.batch_annotate_files(request=request)
    print(response)