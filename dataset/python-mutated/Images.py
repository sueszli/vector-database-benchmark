""" Handling of images, esp. format conversions for icons.

"""
from .FileOperations import getFilenameExtension, hasFilenameExtension
from .Utils import isMacOS, isWin32Windows

def checkIconUsage(logger, icon_path):
    if False:
        return 10
    icon_format = getFilenameExtension(icon_path)
    if icon_format != '.icns' and isMacOS():
        needs_conversion = True
    elif icon_format != '.ico' and isWin32Windows():
        needs_conversion = True
    else:
        needs_conversion = False
    if needs_conversion:
        try:
            import imageio
        except ImportError:
            logger.sysexit("Need to install 'imageio' to let automatically convert the non native icon image (%s) in file in '%s'." % (icon_format[1:].upper(), icon_path))

def convertImageToIconFormat(logger, image_filename, converted_icon_filename):
    if False:
        for i in range(10):
            print('nop')
    'Convert image file to icon file.'
    icon_format = converted_icon_filename.rsplit('.', 1)[1].lower()
    assert hasFilenameExtension(converted_icon_filename, ('.ico', '.icns')), icon_format
    import imageio
    try:
        image = imageio.imread(image_filename)
    except ValueError:
        logger.sysexit("Unsupported file format for 'imageio' in '%s', use e.g. PNG or other supported file formats instead." % image_filename)
    imageio.imwrite(converted_icon_filename, image)