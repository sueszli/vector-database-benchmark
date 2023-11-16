from google.cloud import vision_v1p1beta1

def sample_batch_annotate_images():
    if False:
        for i in range(10):
            print('nop')
    client = vision_v1p1beta1.ImageAnnotatorClient()
    request = vision_v1p1beta1.BatchAnnotateImagesRequest()
    response = client.batch_annotate_images(request=request)
    print(response)