from google.cloud import vision_v1

def sample_async_batch_annotate_images():
    if False:
        for i in range(10):
            print('nop')
    client = vision_v1.ImageAnnotatorClient()
    request = vision_v1.AsyncBatchAnnotateImagesRequest()
    operation = client.async_batch_annotate_images(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)