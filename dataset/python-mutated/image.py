try:
    import cStringIO as StringIO
except ImportError:
    import StringIO
from PIL import Image
from PIL import ImageEnhance
from random import randrange
Image.preinit()
Image._initialized = 2

def image_resize_image(base64_source, size=(1024, 1024), encoding='base64', filetype=None, avoid_if_small=False):
    if False:
        while True:
            i = 10
    " Function to resize an image. The image will be resized to the given\n        size, while keeping the aspect ratios, and holes in the image will be\n        filled with transparent background. The image will not be stretched if\n        smaller than the expected size.\n        Steps of the resizing:\n        - Compute width and height if not specified.\n        - if avoid_if_small: if both image sizes are smaller than the requested\n          sizes, the original image is returned. This is used to avoid adding\n          transparent content around images that we do not want to alter but\n          just resize if too big. This is used for example when storing images\n          in the 'image' field: we keep the original image, resized to a maximal\n          size, without adding transparent content around it if smaller.\n        - create a thumbnail of the source image through using the thumbnail\n          function. Aspect ratios are preserved when using it. Note that if the\n          source image is smaller than the expected size, it will not be\n          extended, but filled to match the size.\n        - create a transparent background that will hold the final image.\n        - paste the thumbnail on the transparent background and center it.\n\n        :param base64_source: base64-encoded version of the source\n            image; if False, returns False\n        :param size: 2-tuple(width, height). A None value for any of width or\n            height mean an automatically computed value based respectivelly\n            on height or width of the source image.\n        :param encoding: the output encoding\n        :param filetype: the output filetype, by default the source image's\n        :type filetype: str, any PIL image format (supported for creation)\n        :param avoid_if_small: do not resize if image height and width\n            are smaller than the expected size.\n    "
    if not base64_source:
        return False
    if size == (None, None):
        return base64_source
    image_stream = StringIO.StringIO(base64_source.decode(encoding))
    image = Image.open(image_stream)
    filetype = (filetype or image.format).upper()
    filetype = {'BMP': 'PNG'}.get(filetype, filetype)
    (asked_width, asked_height) = size
    if asked_width is None:
        asked_width = int(image.size[0] * (float(asked_height) / image.size[1]))
    if asked_height is None:
        asked_height = int(image.size[1] * (float(asked_width) / image.size[0]))
    size = (asked_width, asked_height)
    if avoid_if_small and image.size[0] <= size[0] and (image.size[1] <= size[1]):
        return base64_source
    if image.size != size:
        image = image_resize_and_sharpen(image, size)
    if image.mode not in ['1', 'L', 'P', 'RGB', 'RGBA']:
        image = image.convert('RGB')
    background_stream = StringIO.StringIO()
    image.save(background_stream, filetype)
    return background_stream.getvalue().encode(encoding)

def image_resize_and_sharpen(image, size, preserve_aspect_ratio=False, factor=2.0):
    if False:
        for i in range(10):
            print('nop')
    '\n        Create a thumbnail by resizing while keeping ratio.\n        A sharpen filter is applied for a better looking result.\n\n        :param image: PIL.Image.Image()\n        :param size: 2-tuple(width, height)\n        :param preserve_aspect_ratio: boolean (default: False)\n        :param factor: Sharpen factor (default: 2.0)\n    '
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    image.thumbnail(size, Image.ANTIALIAS)
    if preserve_aspect_ratio:
        size = image.size
    sharpener = ImageEnhance.Sharpness(image)
    resized_image = sharpener.enhance(factor)
    image = Image.new('RGBA', size, (255, 255, 255, 0))
    image.paste(resized_image, ((size[0] - resized_image.size[0]) / 2, (size[1] - resized_image.size[1]) / 2))
    return image

def image_save_for_web(image, fp=None, format=None):
    if False:
        i = 10
        return i + 15
    '\n        Save image optimized for web usage.\n\n        :param image: PIL.Image.Image()\n        :param fp: File name or file object. If not specified, a bytestring is returned.\n        :param format: File format if could not be deduced from image.\n    '
    opt = dict(format=image.format or format)
    if image.format == 'PNG':
        opt.update(optimize=True)
        alpha = False
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            alpha = image.convert('RGBA').split()[-1]
        if image.mode != 'P':
            image = image.convert('RGBA').convert('P', palette=Image.WEB, colors=256)
        if alpha:
            image.putalpha(alpha)
    elif image.format == 'JPEG':
        opt.update(optimize=True, quality=80)
    if fp:
        image.save(fp, **opt)
    else:
        img = StringIO.StringIO()
        image.save(img, **opt)
        return img.getvalue()

def image_resize_image_big(base64_source, size=(1024, 1024), encoding='base64', filetype=None, avoid_if_small=True):
    if False:
        while True:
            i = 10
    " Wrapper on image_resize_image, to resize images larger than the standard\n        'big' image size: 1024x1024px.\n        :param size, encoding, filetype, avoid_if_small: refer to image_resize_image\n    "
    return image_resize_image(base64_source, size, encoding, filetype, avoid_if_small)

def image_resize_image_medium(base64_source, size=(128, 128), encoding='base64', filetype=None, avoid_if_small=False):
    if False:
        while True:
            i = 10
    " Wrapper on image_resize_image, to resize to the standard 'medium'\n        image size: 180x180.\n        :param size, encoding, filetype, avoid_if_small: refer to image_resize_image\n    "
    return image_resize_image(base64_source, size, encoding, filetype, avoid_if_small)

def image_resize_image_small(base64_source, size=(64, 64), encoding='base64', filetype=None, avoid_if_small=False):
    if False:
        return 10
    " Wrapper on image_resize_image, to resize to the standard 'small' image\n        size: 50x50.\n        :param size, encoding, filetype, avoid_if_small: refer to image_resize_image\n    "
    return image_resize_image(base64_source, size, encoding, filetype, avoid_if_small)

def crop_image(data, type='top', ratio=False, thumbnail_ratio=None, image_format='PNG'):
    if False:
        print('Hello World!')
    " Used for cropping image and create thumbnail\n        :param data: base64 data of image.\n        :param type: Used for cropping position possible\n            Possible Values : 'top', 'center', 'bottom'\n        :param ratio: Cropping ratio\n            e.g for (4,3), (16,9), (16,10) etc\n            send ratio(1,1) to generate square image\n        :param thumbnail_ratio: It is size reduce ratio for thumbnail\n            e.g. thumbnail_ratio=2 will reduce your 500x500 image converted in to 250x250\n        :param image_format: return image format PNG,JPEG etc\n    "
    if not data:
        return False
    image_stream = Image.open(StringIO.StringIO(data.decode('base64')))
    output_stream = StringIO.StringIO()
    (w, h) = image_stream.size
    new_h = h
    new_w = w
    if ratio:
        (w_ratio, h_ratio) = ratio
        new_h = w * h_ratio / w_ratio
        new_w = w
        if new_h > h:
            new_h = h
            new_w = h * w_ratio / h_ratio
    if type == 'top':
        cropped_image = image_stream.crop((0, 0, new_w, new_h))
        cropped_image.save(output_stream, format=image_format)
    elif type == 'center':
        cropped_image = image_stream.crop(((w - new_w) / 2, (h - new_h) / 2, (w + new_w) / 2, (h + new_h) / 2))
        cropped_image.save(output_stream, format=image_format)
    elif type == 'bottom':
        cropped_image = image_stream.crop((0, h - new_h, new_w, h))
        cropped_image.save(output_stream, format=image_format)
    else:
        raise ValueError('ERROR: invalid value for crop_type')
    if thumbnail_ratio:
        thumb_image = Image.open(StringIO.StringIO(output_stream.getvalue()))
        thumb_image.thumbnail((new_w / thumbnail_ratio, new_h / thumbnail_ratio), Image.ANTIALIAS)
        thumb_image.save(output_stream, image_format)
    return output_stream.getvalue().encode('base64')

def image_colorize(original, randomize=True, color=(255, 255, 255)):
    if False:
        return 10
    ' Add a color to the transparent background of an image.\n        :param original: file object on the original image file\n        :param randomize: randomize the background color\n        :param color: background-color, if not randomize\n    '
    original = Image.open(StringIO.StringIO(original))
    image = Image.new('RGB', original.size)
    if randomize:
        color = (randrange(32, 224, 24), randrange(32, 224, 24), randrange(32, 224, 24))
    image.paste(color, box=(0, 0) + original.size)
    image.paste(original, mask=original)
    buffer = StringIO.StringIO()
    image.save(buffer, 'PNG')
    return buffer.getvalue()

def image_get_resized_images(base64_source, return_big=False, return_medium=True, return_small=True, big_name='image', medium_name='image_medium', small_name='image_small', avoid_resize_big=True, avoid_resize_medium=False, avoid_resize_small=False):
    if False:
        i = 10
        return i + 15
    " Standard tool function that returns a dictionary containing the\n        big, medium and small versions of the source image. This function\n        is meant to be used for the methods of functional fields for\n        models using images.\n\n        Default parameters are given to be used for the getter of functional\n        image fields,  for example with res.users or res.partner. It returns\n        only image_medium and image_small values, to update those fields.\n\n        :param base64_source: base64-encoded version of the source\n            image; if False, all returnes values will be False\n        :param return_{..}: if set, computes and return the related resizing\n            of the image\n        :param {..}_name: key of the resized image in the return dictionary;\n            'image', 'image_medium' and 'image_small' by default.\n        :param avoid_resize_[..]: see avoid_if_small parameter\n        :return return_dict: dictionary with resized images, depending on\n            previous parameters.\n    "
    return_dict = dict()
    if return_big:
        return_dict[big_name] = image_resize_image_big(base64_source, avoid_if_small=avoid_resize_big)
    if return_medium:
        return_dict[medium_name] = image_resize_image_medium(base64_source, avoid_if_small=avoid_resize_medium)
    if return_small:
        return_dict[small_name] = image_resize_image_small(base64_source, avoid_if_small=avoid_resize_small)
    return return_dict

def image_resize_images(vals, big_name='image', medium_name='image_medium', small_name='image_small'):
    if False:
        while True:
            i = 10
    ' Update ``vals`` with image fields resized as expected. '
    if big_name in vals:
        vals.update(image_get_resized_images(vals[big_name], return_big=True, return_medium=True, return_small=True, big_name=big_name, medium_name=medium_name, small_name=small_name, avoid_resize_big=True, avoid_resize_medium=False, avoid_resize_small=False))
    elif medium_name in vals:
        vals.update(image_get_resized_images(vals[medium_name], return_big=True, return_medium=True, return_small=True, big_name=big_name, medium_name=medium_name, small_name=small_name, avoid_resize_big=True, avoid_resize_medium=True, avoid_resize_small=False))
    elif small_name in vals:
        vals.update(image_get_resized_images(vals[small_name], return_big=True, return_medium=True, return_small=True, big_name=big_name, medium_name=medium_name, small_name=small_name, avoid_resize_big=True, avoid_resize_medium=True, avoid_resize_small=True))
if __name__ == '__main__':
    import sys
    assert len(sys.argv) == 3, 'Usage to Test: image.py SRC.png DEST.png'
    img = file(sys.argv[1], 'rb').read().encode('base64')
    new = image_resize_image(img, (128, 100))
    file(sys.argv[2], 'wb').write(new.decode('base64'))