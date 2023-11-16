from r2.lib.providers.image_resizing import ImageResizingProvider

class UnsplashitImageResizingProvider(ImageResizingProvider):
    """A simple resizer that provides correctly-sized kitten images.

    Useful if you don't want the external dependencies of imgix, but need
    correctly-sized images for testing a UI.
    """

    def resize_image(self, image, width=None, censor_nsfw=False, max_ratio=None):
        if False:
            i = 10
            return i + 15
        if width is None:
            width = image['width']
        height = width * 2
        return 'https://unsplash.it/%d/%d' % (width, height)