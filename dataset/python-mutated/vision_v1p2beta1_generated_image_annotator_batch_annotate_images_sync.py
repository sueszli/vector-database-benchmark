from google.cloud import vision_v1p2beta1

def sample_batch_annotate_images():
    if False:
        while True:
            i = 10
    client = vision_v1p2beta1.ImageAnnotatorClient()
    request = vision_v1p2beta1.BatchAnnotateImagesRequest()
    response = client.batch_annotate_images(request=request)
    print(response)