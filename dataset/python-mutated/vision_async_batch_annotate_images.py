from google.cloud import vision_v1

def sample_async_batch_annotate_images(input_image_uri='gs://cloud-samples-data/vision/label/wakeupcat.jpg', output_uri='gs://your-bucket/prefix/'):
    if False:
        while True:
            i = 10
    'Perform async batch image annotation.'
    client = vision_v1.ImageAnnotatorClient()
    source = {'image_uri': input_image_uri}
    image = {'source': source}
    features = [{'type_': vision_v1.Feature.Type.LABEL_DETECTION}, {'type_': vision_v1.Feature.Type.IMAGE_PROPERTIES}]
    requests = [{'image': image, 'features': features}]
    gcs_destination = {'uri': output_uri}
    batch_size = 2
    output_config = {'gcs_destination': gcs_destination, 'batch_size': batch_size}
    operation = client.async_batch_annotate_images(requests=requests, output_config=output_config)
    print('Waiting for operation to complete...')
    response = operation.result(90)
    gcs_output_uri = response.output_config.gcs_destination.uri
    print(f'Output written to GCS with prefix: {gcs_output_uri}')