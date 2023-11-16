"""Outlines document text given an image.

Example:
    python doctext.py resources/text_menu.jpg
"""
import argparse
from enum import Enum
from google.cloud import vision
from PIL import Image, ImageDraw

class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5

def draw_boxes(image, bounds, color):
    if False:
        return 10
    'Draws a border around the image using the hints in the vector list.\n\n    Args:\n        image: the input image object.\n        bounds: list of coordinates for the boxes.\n        color: the color of the box.\n\n    Returns:\n        An image with colored bounds added.\n    '
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        draw.polygon([bound.vertices[0].x, bound.vertices[0].y, bound.vertices[1].x, bound.vertices[1].y, bound.vertices[2].x, bound.vertices[2].y, bound.vertices[3].x, bound.vertices[3].y], None, color)
    return image

def get_document_bounds(image_file, feature):
    if False:
        for i in range(10):
            print('nop')
    'Finds the document bounds given an image and feature type.\n\n    Args:\n        image_file: path to the image file.\n        feature: feature type to detect.\n\n    Returns:\n        List of coordinates for the corresponding feature type.\n    '
    client = vision.ImageAnnotatorClient()
    bounds = []
    with open(image_file, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    document = response.full_text_annotation
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        if feature == FeatureType.SYMBOL:
                            bounds.append(symbol.bounding_box)
                    if feature == FeatureType.WORD:
                        bounds.append(word.bounding_box)
                if feature == FeatureType.PARA:
                    bounds.append(paragraph.bounding_box)
            if feature == FeatureType.BLOCK:
                bounds.append(block.bounding_box)
    return bounds

def render_doc_text(filein, fileout):
    if False:
        for i in range(10):
            print('nop')
    'Outlines document features (blocks, paragraphs and words) given an image.\n\n    Args:\n        filein: path to the input image.\n        fileout: path to the output image.\n    '
    image = Image.open(filein)
    bounds = get_document_bounds(filein, FeatureType.BLOCK)
    draw_boxes(image, bounds, 'blue')
    bounds = get_document_bounds(filein, FeatureType.PARA)
    draw_boxes(image, bounds, 'red')
    bounds = get_document_bounds(filein, FeatureType.WORD)
    draw_boxes(image, bounds, 'yellow')
    if fileout != 0:
        image.save(fileout)
    else:
        image.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('detect_file', help='The image for text detection.')
    parser.add_argument('-out_file', help='Optional output file', default=0)
    args = parser.parse_args()
    render_doc_text(args.detect_file, args.out_file)