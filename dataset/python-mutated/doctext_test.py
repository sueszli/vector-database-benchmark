import os
import doctext

def test_text() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Checks the output image for drawing the crop hint is created.'
    doctext.render_doc_text('resources/text_menu.jpg', 'output-text.jpg')
    assert os.path.isfile('output-text.jpg')