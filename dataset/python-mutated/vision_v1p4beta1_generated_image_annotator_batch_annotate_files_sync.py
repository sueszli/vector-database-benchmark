from google.cloud import vision_v1p4beta1

def sample_batch_annotate_files():
    if False:
        print('Hello World!')
    client = vision_v1p4beta1.ImageAnnotatorClient()
    request = vision_v1p4beta1.BatchAnnotateFilesRequest()
    response = client.batch_annotate_files(request=request)
    print(response)