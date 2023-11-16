from google.cloud import vision_v1p3beta1

def sample_batch_annotate_images():
    if False:
        i = 10
        return i + 15
    client = vision_v1p3beta1.ImageAnnotatorClient()
    request = vision_v1p3beta1.BatchAnnotateImagesRequest()
    response = client.batch_annotate_images(request=request)
    print(response)