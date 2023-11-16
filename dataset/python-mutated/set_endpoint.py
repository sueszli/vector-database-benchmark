def set_endpoint():
    if False:
        while True:
            i = 10
    'Change your endpoint'
    from google.cloud import vision
    client_options = {'api_endpoint': 'eu-vision.googleapis.com'}
    client = vision.ImageAnnotatorClient(client_options=client_options)
    image_source = vision.ImageSource(image_uri='gs://cloud-samples-data/vision/text/screen.jpg')
    image = vision.Image(source=image_source)
    response = client.text_detection(image=image)
    print('Texts:')
    for text in response.text_annotations:
        print(f'{text.description}')
        vertices = [f'({vertex.x},{vertex.y})' for vertex in text.bounding_poly.vertices]
        print('bounds: {}\n'.format(','.join(vertices)))
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))
if __name__ == '__main__':
    set_endpoint()