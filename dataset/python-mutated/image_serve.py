import hug

@hug.get('/image.png', output=hug.output_format.png_image)
def image():
    if False:
        return 10
    'Serves up a PNG image.'
    return '../artwork/logo.png'