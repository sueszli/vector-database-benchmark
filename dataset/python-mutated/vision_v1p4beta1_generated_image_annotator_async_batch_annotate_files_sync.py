from google.cloud import vision_v1p4beta1

def sample_async_batch_annotate_files():
    if False:
        while True:
            i = 10
    client = vision_v1p4beta1.ImageAnnotatorClient()
    request = vision_v1p4beta1.AsyncBatchAnnotateFilesRequest()
    operation = client.async_batch_annotate_files(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)