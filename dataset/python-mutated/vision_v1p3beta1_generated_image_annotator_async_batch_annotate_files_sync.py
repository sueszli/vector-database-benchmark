from google.cloud import vision_v1p3beta1

def sample_async_batch_annotate_files():
    if False:
        print('Hello World!')
    client = vision_v1p3beta1.ImageAnnotatorClient()
    request = vision_v1p3beta1.AsyncBatchAnnotateFilesRequest()
    operation = client.async_batch_annotate_files(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)