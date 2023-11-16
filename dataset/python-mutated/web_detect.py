"""Demonstrates web detection using the Google Cloud Vision API.

Example usage:
  python web_detect.py https://goo.gl/X4qcB6
  python web_detect.py ../detect/resources/landmark.jpg
  python web_detect.py gs://your-bucket/image.png
"""
import argparse
from google.cloud import vision

def annotate(path: str) -> vision.WebDetection:
    if False:
        i = 10
        return i + 15
    'Returns web annotations given the path to an image.\n\n    Args:\n        path: path to the input image.\n\n    Returns:\n        An WebDetection object with relevant information of the\n        image from the internet (i.e., the annotations).\n    '
    client = vision.ImageAnnotatorClient()
    if path.startswith('http') or path.startswith('gs:'):
        image = vision.Image()
        image.source.image_uri = path
    else:
        with open(path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
    web_detection = client.web_detection(image=image).web_detection
    return web_detection

def report(annotations: vision.WebDetection) -> None:
    if False:
        while True:
            i = 10
    'Prints detected features in the provided web annotations.\n\n    Args:\n        annotations: The web annotations (WebDetection object) from which\n        the features should be parsed and printed.\n    '
    if annotations.pages_with_matching_images:
        print(f'\n{len(annotations.pages_with_matching_images)} Pages with matching images retrieved')
        for page in annotations.pages_with_matching_images:
            print(f'Url   : {page.url}')
    if annotations.full_matching_images:
        print(f'\n{len(annotations.full_matching_images)} Full Matches found: ')
        for image in annotations.full_matching_images:
            print(f'Url  : {image.url}')
    if annotations.partial_matching_images:
        print(f'\n{len(annotations.partial_matching_images)} Partial Matches found: ')
        for image in annotations.partial_matching_images:
            print(f'Url  : {image.url}')
    if annotations.web_entities:
        print(f'\n{len(annotations.web_entities)} Web entities found: ')
        for entity in annotations.web_entities:
            print(f'Score      : {entity.score}')
            print(f'Description: {entity.description}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    path_help = str('The image to detect, can be web URI, Google Cloud Storage, or path to local file.')
    parser.add_argument('image_url', help=path_help)
    args = parser.parse_args()
    report(annotate(args.image_url))