"""
Caption module
"""
try:
    from PIL import Image
    PIL = True
except ImportError:
    PIL = False
from ..hfpipeline import HFPipeline

class Caption(HFPipeline):
    """
    Constructs captions for images.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, **kwargs):
        if False:
            print('Hello World!')
        if not PIL:
            raise ImportError('Captions pipeline is not available - install "pipeline" extra to enable')
        super().__init__('image-to-text', path, quantize, gpu, model, **kwargs)

    def __call__(self, images):
        if False:
            for i in range(10):
                print('nop')
        '\n        Builds captions for images.\n\n        This method supports a single image or a list of images. If the input is an image, the return\n        type is a string. If text is a list, a list of strings is returned\n\n        Args:\n            images: image|list\n\n        Returns:\n            list of captions\n        '
        values = [images] if not isinstance(images, list) else images
        values = [Image.open(image) if isinstance(image, str) else image for image in values]
        captions = []
        for result in self.pipeline(values):
            text = ' '.join([r['generated_text'] for r in result]).strip()
            captions.append(text)
        return captions[0] if not isinstance(images, list) else captions