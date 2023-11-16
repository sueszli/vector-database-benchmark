"""Outputs a cropped image or an image highlighting crop regions on an image.

Examples:
    python crop_hints.py resources/cropme.jpg draw
    python crop_hints.py resources/cropme.jpg crop
"""
import argparse
from typing import MutableSequence
from google.cloud import vision
from PIL import Image, ImageDraw

def get_crop_hint(path: str) -> MutableSequence[vision.Vertex]:
    if False:
        for i in range(10):
            print('nop')
    'Detect crop hints on a single image and return the first result.\n\n    Args:\n        path: path to the image file.\n\n    Returns:\n        The vertices for the bounding polygon.\n    '
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    crop_hints_params = vision.CropHintsParams(aspect_ratios=[1.77])
    image_context = vision.ImageContext(crop_hints_params=crop_hints_params)
    response = client.crop_hints(image=image, image_context=image_context)
    hints = response.crop_hints_annotation.crop_hints
    vertices = hints[0].bounding_poly.vertices
    return vertices

def draw_hint(image_file: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Draw a border around the image using the hints in the vector list.\n\n    Args:\n        image_file: path to the image file.\n    '
    vects = get_crop_hint(image_file)
    im = Image.open(image_file)
    draw = ImageDraw.Draw(im)
    draw.polygon([vects[0].x, vects[0].y, vects[1].x, vects[1].y, vects[2].x, vects[2].y, vects[3].x, vects[3].y], None, 'red')
    im.save('output-hint.jpg', 'JPEG')
    print('Saved new image to output-hint.jpg')

def crop_to_hint(image_file: str) -> None:
    if False:
        print('Hello World!')
    'Crop the image using the hints in the vector list.\n\n    Args:\n        image_file: path to the image file.\n    '
    vects = get_crop_hint(image_file)
    im = Image.open(image_file)
    im2 = im.crop([vects[0].x, vects[0].y, vects[2].x - 1, vects[2].y - 1])
    im2.save('output-crop.jpg', 'JPEG')
    print('Saved new image to output-crop.jpg')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_file', help="The image you'd like to crop.")
    parser.add_argument('mode', help='Set to "crop" or "draw".')
    args = parser.parse_args()
    if args.mode == 'crop':
        crop_to_hint(args.image_file)
    elif args.mode == 'draw':
        draw_hint(args.image_file)