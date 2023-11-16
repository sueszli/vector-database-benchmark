from functools import update_wrapper
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
import click

@click.group(chain=True)
def cli():
    if False:
        return 10
    'This script processes a bunch of images through pillow in a unix\n    pipe.  One commands feeds into the next.\n\n    Example:\n\n    \x08\n        imagepipe open -i example01.jpg resize -w 128 display\n        imagepipe open -i example02.jpg blur save\n    '

@cli.result_callback()
def process_commands(processors):
    if False:
        return 10
    'This result callback is invoked with an iterable of all the chained\n    subcommands.  As in this example each subcommand returns a function\n    we can chain them together to feed one into the other, similar to how\n    a pipe on unix works.\n    '
    stream = ()
    for processor in processors:
        stream = processor(stream)
    for _ in stream:
        pass

def processor(f):
    if False:
        while True:
            i = 10
    'Helper decorator to rewrite a function so that it returns another\n    function from it.\n    '

    def new_func(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')

        def processor(stream):
            if False:
                return 10
            return f(stream, *args, **kwargs)
        return processor
    return update_wrapper(new_func, f)

def generator(f):
    if False:
        while True:
            i = 10
    'Similar to the :func:`processor` but passes through old values\n    unchanged and does not pass through the values as parameter.\n    '

    @processor
    def new_func(stream, *args, **kwargs):
        if False:
            while True:
                i = 10
        yield from stream
        yield from f(*args, **kwargs)
    return update_wrapper(new_func, f)

def copy_filename(new, old):
    if False:
        return 10
    new.filename = old.filename
    return new

@cli.command('open')
@click.option('-i', '--image', 'images', type=click.Path(), multiple=True, help='The image file to open.')
@generator
def open_cmd(images):
    if False:
        return 10
    'Loads one or multiple images for processing.  The input parameter\n    can be specified multiple times to load more than one image.\n    '
    for image in images:
        try:
            click.echo(f"Opening '{image}'")
            if image == '-':
                img = Image.open(click.get_binary_stdin())
                img.filename = '-'
            else:
                img = Image.open(image)
            yield img
        except Exception as e:
            click.echo(f"Could not open image '{image}': {e}", err=True)

@cli.command('save')
@click.option('--filename', default='processed-{:04}.png', type=click.Path(), help='The format for the filename.', show_default=True)
@processor
def save_cmd(images, filename):
    if False:
        return 10
    'Saves all processed images to a series of files.'
    for (idx, image) in enumerate(images):
        try:
            fn = filename.format(idx + 1)
            click.echo(f"Saving '{image.filename}' as '{fn}'")
            yield image.save(fn)
        except Exception as e:
            click.echo(f"Could not save image '{image.filename}': {e}", err=True)

@cli.command('display')
@processor
def display_cmd(images):
    if False:
        while True:
            i = 10
    'Opens all images in an image viewer.'
    for image in images:
        click.echo(f"Displaying '{image.filename}'")
        image.show()
        yield image

@cli.command('resize')
@click.option('-w', '--width', type=int, help='The new width of the image.')
@click.option('-h', '--height', type=int, help='The new height of the image.')
@processor
def resize_cmd(images, width, height):
    if False:
        i = 10
        return i + 15
    'Resizes an image by fitting it into the box without changing\n    the aspect ratio.\n    '
    for image in images:
        (w, h) = (width or image.size[0], height or image.size[1])
        click.echo(f"Resizing '{image.filename}' to {w}x{h}")
        image.thumbnail((w, h))
        yield image

@cli.command('crop')
@click.option('-b', '--border', type=int, help='Crop the image from all sides by this amount.')
@processor
def crop_cmd(images, border):
    if False:
        while True:
            i = 10
    'Crops an image from all edges.'
    for image in images:
        box = [0, 0, image.size[0], image.size[1]]
        if border is not None:
            for (idx, val) in enumerate(box):
                box[idx] = max(0, val - border)
            click.echo(f"Cropping '{image.filename}' by {border}px")
            yield copy_filename(image.crop(box), image)
        else:
            yield image

def convert_rotation(ctx, param, value):
    if False:
        for i in range(10):
            print('nop')
    if value is None:
        return
    value = value.lower()
    if value in ('90', 'r', 'right'):
        return (Image.ROTATE_90, 90)
    if value in ('180', '-180'):
        return (Image.ROTATE_180, 180)
    if value in ('-90', '270', 'l', 'left'):
        return (Image.ROTATE_270, 270)
    raise click.BadParameter(f"invalid rotation '{value}'")

def convert_flip(ctx, param, value):
    if False:
        i = 10
        return i + 15
    if value is None:
        return
    value = value.lower()
    if value in ('lr', 'leftright'):
        return (Image.FLIP_LEFT_RIGHT, 'left to right')
    if value in ('tb', 'topbottom', 'upsidedown', 'ud'):
        return (Image.FLIP_LEFT_RIGHT, 'top to bottom')
    raise click.BadParameter(f"invalid flip '{value}'")

@cli.command('transpose')
@click.option('-r', '--rotate', callback=convert_rotation, help='Rotates the image (in degrees)')
@click.option('-f', '--flip', callback=convert_flip, help='Flips the image  [LR / TB]')
@processor
def transpose_cmd(images, rotate, flip):
    if False:
        while True:
            i = 10
    'Transposes an image by either rotating or flipping it.'
    for image in images:
        if rotate is not None:
            (mode, degrees) = rotate
            click.echo(f"Rotate '{image.filename}' by {degrees}deg")
            image = copy_filename(image.transpose(mode), image)
        if flip is not None:
            (mode, direction) = flip
            click.echo(f"Flip '{image.filename}' {direction}")
            image = copy_filename(image.transpose(mode), image)
        yield image

@cli.command('blur')
@click.option('-r', '--radius', default=2, show_default=True, help='The blur radius.')
@processor
def blur_cmd(images, radius):
    if False:
        while True:
            i = 10
    'Applies gaussian blur.'
    blur = ImageFilter.GaussianBlur(radius)
    for image in images:
        click.echo(f"Blurring '{image.filename}' by {radius}px")
        yield copy_filename(image.filter(blur), image)

@cli.command('smoothen')
@click.option('-i', '--iterations', default=1, show_default=True, help='How many iterations of the smoothen filter to run.')
@processor
def smoothen_cmd(images, iterations):
    if False:
        for i in range(10):
            print('nop')
    'Applies a smoothening filter.'
    for image in images:
        click.echo(f"Smoothening {image.filename!r} {iterations} time{('s' if iterations != 1 else '')}")
        for _ in range(iterations):
            image = copy_filename(image.filter(ImageFilter.BLUR), image)
        yield image

@cli.command('emboss')
@processor
def emboss_cmd(images):
    if False:
        while True:
            i = 10
    'Embosses an image.'
    for image in images:
        click.echo(f"Embossing '{image.filename}'")
        yield copy_filename(image.filter(ImageFilter.EMBOSS), image)

@cli.command('sharpen')
@click.option('-f', '--factor', default=2.0, help='Sharpens the image.', show_default=True)
@processor
def sharpen_cmd(images, factor):
    if False:
        return 10
    'Sharpens an image.'
    for image in images:
        click.echo(f"Sharpen '{image.filename}' by {factor}")
        enhancer = ImageEnhance.Sharpness(image)
        yield copy_filename(enhancer.enhance(max(1.0, factor)), image)

@cli.command('paste')
@click.option('-l', '--left', default=0, help='Offset from left.')
@click.option('-r', '--right', default=0, help='Offset from right.')
@processor
def paste_cmd(images, left, right):
    if False:
        print('Hello World!')
    'Pastes the second image on the first image and leaves the rest\n    unchanged.\n    '
    imageiter = iter(images)
    image = next(imageiter, None)
    to_paste = next(imageiter, None)
    if to_paste is None:
        if image is not None:
            yield image
        return
    click.echo(f"Paste '{to_paste.filename}' on '{image.filename}'")
    mask = None
    if to_paste.mode == 'RGBA' or 'transparency' in to_paste.info:
        mask = to_paste
    image.paste(to_paste, (left, right), mask)
    image.filename += f'+{to_paste.filename}'
    yield image
    yield from imageiter