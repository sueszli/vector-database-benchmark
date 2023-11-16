"""This application demonstrates how to perform basic operations with the
Google Cloud Vision API.

Example Usage:
python detect.py text ./resources/wakeupcat.jpg
python detect.py labels ./resources/landmark.jpg
python detect.py web ./resources/landmark.jpg
python detect.py web-uri http://wheresgus.com/dog.JPG
python detect.py web-geo ./resources/city.jpg
python detect.py faces-uri gs://your-bucket/file.jpg
python detect.py ocr-uri gs://python-docs-samples-tests/HodgeConj.pdf gs://BUCKET_NAME/PREFIX/
python detect.py object-localization ./resources/puppies.jpg
python detect.py object-localization-uri gs://...

For more information, the documentation at
https://cloud.google.com/vision/docs.
"""
import argparse

def detect_faces(path):
    if False:
        while True:
            i = 10
    'Detects faces in an image.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.face_detection(image=image)
    faces = response.face_annotations
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY')
    print('Faces:')
    for face in faces:
        print(f'anger: {likelihood_name[face.anger_likelihood]}')
        print(f'joy: {likelihood_name[face.joy_likelihood]}')
        print(f'surprise: {likelihood_name[face.surprise_likelihood]}')
        vertices = [f'({vertex.x},{vertex.y})' for vertex in face.bounding_poly.vertices]
        print('face bounds: {}'.format(','.join(vertices)))
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def detect_faces_uri(uri):
    if False:
        print('Hello World!')
    'Detects faces in the file located in Google Cloud Storage or the web.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri
    response = client.face_detection(image=image)
    faces = response.face_annotations
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY')
    print('Faces:')
    for face in faces:
        print(f'anger: {likelihood_name[face.anger_likelihood]}')
        print(f'joy: {likelihood_name[face.joy_likelihood]}')
        print(f'surprise: {likelihood_name[face.surprise_likelihood]}')
        vertices = [f'({vertex.x},{vertex.y})' for vertex in face.bounding_poly.vertices]
        print('face bounds: {}'.format(','.join(vertices)))
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def detect_labels(path):
    if False:
        i = 10
        return i + 15
    'Detects labels in the file.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations
    print('Labels:')
    for label in labels:
        print(label.description)
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def detect_labels_uri(uri):
    if False:
        i = 10
        return i + 15
    'Detects labels in the file located in Google Cloud Storage or on the\n    Web.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri
    response = client.label_detection(image=image)
    labels = response.label_annotations
    print('Labels:')
    for label in labels:
        print(label.description)
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def detect_landmarks(path):
    if False:
        for i in range(10):
            print('nop')
    'Detects landmarks in the file.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.landmark_detection(image=image)
    landmarks = response.landmark_annotations
    print('Landmarks:')
    for landmark in landmarks:
        print(landmark.description)
        for location in landmark.locations:
            lat_lng = location.lat_lng
            print(f'Latitude {lat_lng.latitude}')
            print(f'Longitude {lat_lng.longitude}')
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def detect_landmarks_uri(uri):
    if False:
        for i in range(10):
            print('nop')
    'Detects landmarks in the file located in Google Cloud Storage or on the\n    Web.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri
    response = client.landmark_detection(image=image)
    landmarks = response.landmark_annotations
    print('Landmarks:')
    for landmark in landmarks:
        print(landmark.description)
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def detect_logos(path):
    if False:
        while True:
            i = 10
    'Detects logos in the file.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.logo_detection(image=image)
    logos = response.logo_annotations
    print('Logos:')
    for logo in logos:
        print(logo.description)
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def detect_logos_uri(uri):
    if False:
        return 10
    'Detects logos in the file located in Google Cloud Storage or on the Web.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri
    response = client.logo_detection(image=image)
    logos = response.logo_annotations
    print('Logos:')
    for logo in logos:
        print(logo.description)
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def detect_safe_search(path):
    if False:
        return 10
    'Detects unsafe features in the file.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.safe_search_detection(image=image)
    safe = response.safe_search_annotation
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY')
    print('Safe search:')
    print(f'adult: {likelihood_name[safe.adult]}')
    print(f'medical: {likelihood_name[safe.medical]}')
    print(f'spoofed: {likelihood_name[safe.spoof]}')
    print(f'violence: {likelihood_name[safe.violence]}')
    print(f'racy: {likelihood_name[safe.racy]}')
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def detect_safe_search_uri(uri):
    if False:
        while True:
            i = 10
    'Detects unsafe features in the file located in Google Cloud Storage or\n    on the Web.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri
    response = client.safe_search_detection(image=image)
    safe = response.safe_search_annotation
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY')
    print('Safe search:')
    print(f'adult: {likelihood_name[safe.adult]}')
    print(f'medical: {likelihood_name[safe.medical]}')
    print(f'spoofed: {likelihood_name[safe.spoof]}')
    print(f'violence: {likelihood_name[safe.violence]}')
    print(f'racy: {likelihood_name[safe.racy]}')
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def detect_text(path):
    if False:
        return 10
    'Detects text in the file.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')
    for text in texts:
        print(f'\n"{text.description}"')
        vertices = [f'({vertex.x},{vertex.y})' for vertex in text.bounding_poly.vertices]
        print('bounds: {}'.format(','.join(vertices)))
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def detect_text_uri(uri):
    if False:
        return 10
    'Detects text in the file located in Google Cloud Storage or on the Web.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri
    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')
    for text in texts:
        print(f'\n"{text.description}"')
        vertices = [f'({vertex.x},{vertex.y})' for vertex in text.bounding_poly.vertices]
        print('bounds: {}'.format(','.join(vertices)))
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def detect_properties(path):
    if False:
        while True:
            i = 10
    'Detects image properties in the file.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.image_properties(image=image)
    props = response.image_properties_annotation
    print('Properties:')
    for color in props.dominant_colors.colors:
        print(f'fraction: {color.pixel_fraction}')
        print(f'\tr: {color.color.red}')
        print(f'\tg: {color.color.green}')
        print(f'\tb: {color.color.blue}')
        print(f'\ta: {color.color.alpha}')
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def detect_properties_uri(uri):
    if False:
        i = 10
        return i + 15
    'Detects image properties in the file located in Google Cloud Storage or\n    on the Web.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri
    response = client.image_properties(image=image)
    props = response.image_properties_annotation
    print('Properties:')
    for color in props.dominant_colors.colors:
        print(f'frac: {color.pixel_fraction}')
        print(f'\tr: {color.color.red}')
        print(f'\tg: {color.color.green}')
        print(f'\tb: {color.color.blue}')
        print(f'\ta: {color.color.alpha}')
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def detect_web(path):
    if False:
        print('Hello World!')
    'Detects web annotations given an image.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.web_detection(image=image)
    annotations = response.web_detection
    if annotations.best_guess_labels:
        for label in annotations.best_guess_labels:
            print(f'\nBest guess label: {label.label}')
    if annotations.pages_with_matching_images:
        print('\n{} Pages with matching images found:'.format(len(annotations.pages_with_matching_images)))
        for page in annotations.pages_with_matching_images:
            print(f'\n\tPage url   : {page.url}')
            if page.full_matching_images:
                print('\t{} Full Matches found: '.format(len(page.full_matching_images)))
                for image in page.full_matching_images:
                    print(f'\t\tImage url  : {image.url}')
            if page.partial_matching_images:
                print('\t{} Partial Matches found: '.format(len(page.partial_matching_images)))
                for image in page.partial_matching_images:
                    print(f'\t\tImage url  : {image.url}')
    if annotations.web_entities:
        print('\n{} Web entities found: '.format(len(annotations.web_entities)))
        for entity in annotations.web_entities:
            print(f'\n\tScore      : {entity.score}')
            print(f'\tDescription: {entity.description}')
    if annotations.visually_similar_images:
        print('\n{} visually similar images found:\n'.format(len(annotations.visually_similar_images)))
        for image in annotations.visually_similar_images:
            print(f'\tImage url    : {image.url}')
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def detect_web_uri(uri):
    if False:
        print('Hello World!')
    'Detects web annotations in the file located in Google Cloud Storage.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri
    response = client.web_detection(image=image)
    annotations = response.web_detection
    if annotations.best_guess_labels:
        for label in annotations.best_guess_labels:
            print(f'\nBest guess label: {label.label}')
    if annotations.pages_with_matching_images:
        print('\n{} Pages with matching images found:'.format(len(annotations.pages_with_matching_images)))
        for page in annotations.pages_with_matching_images:
            print(f'\n\tPage url   : {page.url}')
            if page.full_matching_images:
                print('\t{} Full Matches found: '.format(len(page.full_matching_images)))
                for image in page.full_matching_images:
                    print(f'\t\tImage url  : {image.url}')
            if page.partial_matching_images:
                print('\t{} Partial Matches found: '.format(len(page.partial_matching_images)))
                for image in page.partial_matching_images:
                    print(f'\t\tImage url  : {image.url}')
    if annotations.web_entities:
        print('\n{} Web entities found: '.format(len(annotations.web_entities)))
        for entity in annotations.web_entities:
            print(f'\n\tScore      : {entity.score}')
            print(f'\tDescription: {entity.description}')
    if annotations.visually_similar_images:
        print('\n{} visually similar images found:\n'.format(len(annotations.visually_similar_images)))
        for image in annotations.visually_similar_images:
            print(f'\tImage url    : {image.url}')
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def web_entities_include_geo_results(path):
    if False:
        print('Hello World!')
    'Detects web annotations given an image, using the geotag metadata\n    in the image to detect web entities.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    web_detection_params = vision.WebDetectionParams(include_geo_results=True)
    image_context = vision.ImageContext(web_detection_params=web_detection_params)
    response = client.web_detection(image=image, image_context=image_context)
    for entity in response.web_detection.web_entities:
        print(f'\n\tScore      : {entity.score}')
        print(f'\tDescription: {entity.description}')
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def web_entities_include_geo_results_uri(uri):
    if False:
        for i in range(10):
            print('nop')
    'Detects web annotations given an image in the file located in\n    Google Cloud Storage., using the geotag metadata in the image to\n    detect web entities.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri
    web_detection_params = vision.WebDetectionParams(include_geo_results=True)
    image_context = vision.ImageContext(web_detection_params=web_detection_params)
    response = client.web_detection(image=image, image_context=image_context)
    for entity in response.web_detection.web_entities:
        print(f'\n\tScore      : {entity.score}')
        print(f'\tDescription: {entity.description}')
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def detect_crop_hints(path):
    if False:
        return 10
    'Detects crop hints in an image.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    crop_hints_params = vision.CropHintsParams(aspect_ratios=[1.77])
    image_context = vision.ImageContext(crop_hints_params=crop_hints_params)
    response = client.crop_hints(image=image, image_context=image_context)
    hints = response.crop_hints_annotation.crop_hints
    for (n, hint) in enumerate(hints):
        print(f'\nCrop Hint: {n}')
        vertices = [f'({vertex.x},{vertex.y})' for vertex in hint.bounding_poly.vertices]
        print('bounds: {}'.format(','.join(vertices)))
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def detect_crop_hints_uri(uri):
    if False:
        i = 10
        return i + 15
    'Detects crop hints in the file located in Google Cloud Storage.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri
    crop_hints_params = vision.CropHintsParams(aspect_ratios=[1.77])
    image_context = vision.ImageContext(crop_hints_params=crop_hints_params)
    response = client.crop_hints(image=image, image_context=image_context)
    hints = response.crop_hints_annotation.crop_hints
    for (n, hint) in enumerate(hints):
        print(f'\nCrop Hint: {n}')
        vertices = [f'({vertex.x},{vertex.y})' for vertex in hint.bounding_poly.vertices]
        print('bounds: {}'.format(','.join(vertices)))
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def detect_document(path):
    if False:
        i = 10
        return i + 15
    'Detects document features in an image.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            print(f'\nBlock confidence: {block.confidence}\n')
            for paragraph in block.paragraphs:
                print('Paragraph confidence: {}'.format(paragraph.confidence))
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    print('Word text: {} (confidence: {})'.format(word_text, word.confidence))
                    for symbol in word.symbols:
                        print('\tSymbol: {} (confidence: {})'.format(symbol.text, symbol.confidence))
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def detect_document_uri(uri):
    if False:
        i = 10
        return i + 15
    'Detects document features in the file located in Google Cloud\n    Storage.'
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri
    response = client.document_text_detection(image=image)
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            print(f'\nBlock confidence: {block.confidence}\n')
            for paragraph in block.paragraphs:
                print('Paragraph confidence: {}'.format(paragraph.confidence))
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    print('Word text: {} (confidence: {})'.format(word_text, word.confidence))
                    for symbol in word.symbols:
                        print('\tSymbol: {} (confidence: {})'.format(symbol.text, symbol.confidence))
    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(response.error.message))

def async_detect_document(gcs_source_uri, gcs_destination_uri):
    if False:
        while True:
            i = 10
    'OCR with PDF/TIFF as source files on GCS'
    import json
    import re
    from google.cloud import vision
    from google.cloud import storage
    mime_type = 'application/pdf'
    batch_size = 2
    client = vision.ImageAnnotatorClient()
    feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
    gcs_source = vision.GcsSource(uri=gcs_source_uri)
    input_config = vision.InputConfig(gcs_source=gcs_source, mime_type=mime_type)
    gcs_destination = vision.GcsDestination(uri=gcs_destination_uri)
    output_config = vision.OutputConfig(gcs_destination=gcs_destination, batch_size=batch_size)
    async_request = vision.AsyncAnnotateFileRequest(features=[feature], input_config=input_config, output_config=output_config)
    operation = client.async_batch_annotate_files(requests=[async_request])
    print('Waiting for the operation to finish.')
    operation.result(timeout=420)
    storage_client = storage.Client()
    match = re.match('gs://([^/]+)/(.+)', gcs_destination_uri)
    bucket_name = match.group(1)
    prefix = match.group(2)
    bucket = storage_client.get_bucket(bucket_name)
    blob_list = [blob for blob in list(bucket.list_blobs(prefix=prefix)) if not blob.name.endswith('/')]
    print('Output files:')
    for blob in blob_list:
        print(blob.name)
    output = blob_list[0]
    json_string = output.download_as_bytes().decode('utf-8')
    response = json.loads(json_string)
    first_page_response = response['responses'][0]
    annotation = first_page_response['fullTextAnnotation']
    print('Full text:\n')
    print(annotation['text'])

def localize_objects(path):
    if False:
        i = 10
        return i + 15
    'Localize objects in the local image.\n\n    Args:\n    path: The path to the local file.\n    '
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    objects = client.object_localization(image=image).localized_object_annotations
    print(f'Number of objects found: {len(objects)}')
    for object_ in objects:
        print(f'\n{object_.name} (confidence: {object_.score})')
        print('Normalized bounding polygon vertices: ')
        for vertex in object_.bounding_poly.normalized_vertices:
            print(f' - ({vertex.x}, {vertex.y})')

def localize_objects_uri(uri):
    if False:
        for i in range(10):
            print('nop')
    'Localize objects in the image on Google Cloud Storage\n\n    Args:\n    uri: The path to the file in Google Cloud Storage (gs://...)\n    '
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri
    objects = client.object_localization(image=image).localized_object_annotations
    print(f'Number of objects found: {len(objects)}')
    for object_ in objects:
        print(f'\n{object_.name} (confidence: {object_.score})')
        print('Normalized bounding polygon vertices: ')
        for vertex in object_.bounding_poly.normalized_vertices:
            print(f' - ({vertex.x}, {vertex.y})')

def run_local(args):
    if False:
        print('Hello World!')
    if args.command == 'faces':
        detect_faces(args.path)
    elif args.command == 'labels':
        detect_labels(args.path)
    elif args.command == 'landmarks':
        detect_landmarks(args.path)
    elif args.command == 'text':
        detect_text(args.path)
    elif args.command == 'logos':
        detect_logos(args.path)
    elif args.command == 'safe-search':
        detect_safe_search(args.path)
    elif args.command == 'properties':
        detect_properties(args.path)
    elif args.command == 'web':
        detect_web(args.path)
    elif args.command == 'crophints':
        detect_crop_hints(args.path)
    elif args.command == 'document':
        detect_document(args.path)
    elif args.command == 'web-geo':
        web_entities_include_geo_results(args.path)
    elif args.command == 'object-localization':
        localize_objects(args.path)

def run_uri(args):
    if False:
        i = 10
        return i + 15
    if args.command == 'text-uri':
        detect_text_uri(args.uri)
    elif args.command == 'faces-uri':
        detect_faces_uri(args.uri)
    elif args.command == 'labels-uri':
        detect_labels_uri(args.uri)
    elif args.command == 'landmarks-uri':
        detect_landmarks_uri(args.uri)
    elif args.command == 'logos-uri':
        detect_logos_uri(args.uri)
    elif args.command == 'safe-search-uri':
        detect_safe_search_uri(args.uri)
    elif args.command == 'properties-uri':
        detect_properties_uri(args.uri)
    elif args.command == 'web-uri':
        detect_web_uri(args.uri)
    elif args.command == 'crophints-uri':
        detect_crop_hints_uri(args.uri)
    elif args.command == 'document-uri':
        detect_document_uri(args.uri)
    elif args.command == 'web-geo-uri':
        web_entities_include_geo_results_uri(args.uri)
    elif args.command == 'ocr-uri':
        async_detect_document(args.uri, args.destination_uri)
    elif args.command == 'object-localization-uri':
        localize_objects_uri(args.uri)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    detect_faces_parser = subparsers.add_parser('faces', help=detect_faces.__doc__)
    detect_faces_parser.add_argument('path')
    faces_file_parser = subparsers.add_parser('faces-uri', help=detect_faces_uri.__doc__)
    faces_file_parser.add_argument('uri')
    detect_labels_parser = subparsers.add_parser('labels', help=detect_labels.__doc__)
    detect_labels_parser.add_argument('path')
    labels_file_parser = subparsers.add_parser('labels-uri', help=detect_labels_uri.__doc__)
    labels_file_parser.add_argument('uri')
    detect_landmarks_parser = subparsers.add_parser('landmarks', help=detect_landmarks.__doc__)
    detect_landmarks_parser.add_argument('path')
    landmark_file_parser = subparsers.add_parser('landmarks-uri', help=detect_landmarks_uri.__doc__)
    landmark_file_parser.add_argument('uri')
    detect_text_parser = subparsers.add_parser('text', help=detect_text.__doc__)
    detect_text_parser.add_argument('path')
    text_file_parser = subparsers.add_parser('text-uri', help=detect_text_uri.__doc__)
    text_file_parser.add_argument('uri')
    detect_logos_parser = subparsers.add_parser('logos', help=detect_logos.__doc__)
    detect_logos_parser.add_argument('path')
    logos_file_parser = subparsers.add_parser('logos-uri', help=detect_logos_uri.__doc__)
    logos_file_parser.add_argument('uri')
    safe_search_parser = subparsers.add_parser('safe-search', help=detect_safe_search.__doc__)
    safe_search_parser.add_argument('path')
    safe_search_file_parser = subparsers.add_parser('safe-search-uri', help=detect_safe_search_uri.__doc__)
    safe_search_file_parser.add_argument('uri')
    properties_parser = subparsers.add_parser('properties', help=detect_properties.__doc__)
    properties_parser.add_argument('path')
    properties_file_parser = subparsers.add_parser('properties-uri', help=detect_properties_uri.__doc__)
    properties_file_parser.add_argument('uri')
    web_parser = subparsers.add_parser('web', help=detect_web.__doc__)
    web_parser.add_argument('path')
    web_uri_parser = subparsers.add_parser('web-uri', help=detect_web_uri.__doc__)
    web_uri_parser.add_argument('uri')
    web_geo_parser = subparsers.add_parser('web-geo', help=web_entities_include_geo_results.__doc__)
    web_geo_parser.add_argument('path')
    web_geo_uri_parser = subparsers.add_parser('web-geo-uri', help=web_entities_include_geo_results_uri.__doc__)
    web_geo_uri_parser.add_argument('uri')
    crop_hints_parser = subparsers.add_parser('crophints', help=detect_crop_hints.__doc__)
    crop_hints_parser.add_argument('path')
    crop_hints_uri_parser = subparsers.add_parser('crophints-uri', help=detect_crop_hints_uri.__doc__)
    crop_hints_uri_parser.add_argument('uri')
    document_parser = subparsers.add_parser('document', help=detect_document.__doc__)
    document_parser.add_argument('path')
    document_uri_parser = subparsers.add_parser('document-uri', help=detect_document_uri.__doc__)
    document_uri_parser.add_argument('uri')
    ocr_uri_parser = subparsers.add_parser('ocr-uri', help=async_detect_document.__doc__)
    ocr_uri_parser.add_argument('uri')
    ocr_uri_parser.add_argument('destination_uri')
    object_localization_parser = subparsers.add_parser('object-localization', help=async_detect_document.__doc__)
    object_localization_parser.add_argument('path')
    object_localization_uri_parser = subparsers.add_parser('object-localization-uri', help=async_detect_document.__doc__)
    object_localization_uri_parser.add_argument('uri')
    args = parser.parse_args()
    if 'uri' in args.command:
        run_uri(args)
    else:
        run_local(args)