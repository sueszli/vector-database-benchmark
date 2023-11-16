from wagtail.images.models import SourceImageIOError

def get_rendition_or_not_found(image, specs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tries to get / create the rendition for the image or renders a not-found image if it does not exist.\n\n    :param image: AbstractImage\n    :param specs: str or Filter\n    :return: Rendition\n    '
    try:
        return image.get_rendition(specs)
    except SourceImageIOError:
        Rendition = image.renditions.model
        rendition = Rendition(image=image, width=0, height=0)
        rendition.file.name = 'not-found'
        return rendition

def get_renditions_or_not_found(image, specs):
    if False:
        while True:
            i = 10
    '\n    Like get_rendition_or_not_found, but for multiple renditions.\n    Tries to get / create the renditions for the image or renders not-found images if the image does not exist.\n\n    :param image: AbstractImage\n    :param specs: iterable of str or Filter\n    '
    try:
        return image.get_renditions(*specs)
    except SourceImageIOError:
        Rendition = image.renditions.model
        rendition = Rendition(image=image, width=0, height=0)
        rendition.file.name = 'not-found'
        return {spec if isinstance(spec, str) else spec.spec: rendition for spec in specs}