import os
import crop_hints

def test_crop() -> None:
    if False:
        while True:
            i = 10
    'Checks the output image for cropping the image is created.'
    file_name = os.path.join(os.path.dirname(__file__), 'resources/cropme.jpg')
    crop_hints.crop_to_hint(file_name)
    assert os.path.isfile('output-crop.jpg')

def test_draw() -> None:
    if False:
        i = 10
        return i + 15
    'Checks the output image for drawing the crop hint is created.'
    file_name = os.path.join(os.path.dirname(__file__), 'resources/cropme.jpg')
    crop_hints.draw_hint(file_name)
    assert os.path.isfile('output-hint.jpg')