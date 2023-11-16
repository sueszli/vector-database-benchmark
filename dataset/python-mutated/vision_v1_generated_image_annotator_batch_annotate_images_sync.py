from google.cloud import vision_v1

def sample_batch_annotate_images():
    if False:
        print('Hello World!')
    client = vision_v1.ImageAnnotatorClient()
    request = vision_v1.BatchAnnotateImagesRequest()
    response = client.batch_annotate_images(request=request)
    print(response)