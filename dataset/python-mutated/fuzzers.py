import io
import warnings
from PIL import Image, ImageDraw, ImageFile, ImageFilter, ImageFont

def enable_decompressionbomb_error():
    if False:
        while True:
            i = 10
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    warnings.filterwarnings('ignore')
    warnings.simplefilter('error', Image.DecompressionBombWarning)

def disable_decompressionbomb_error():
    if False:
        print('Hello World!')
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    warnings.resetwarnings()

def fuzz_image(data):
    if False:
        for i in range(10):
            print('nop')
    with Image.open(io.BytesIO(data)) as im:
        im.rotate(45)
        im.filter(ImageFilter.DETAIL)
        im.save(io.BytesIO(), 'BMP')

def fuzz_font(data):
    if False:
        return 10
    wrapper = io.BytesIO(data)
    try:
        font = ImageFont.truetype(wrapper)
    except OSError:
        return
    font.getbbox('ABC')
    font.getmask('test text')
    with Image.new(mode='RGBA', size=(200, 200)) as im:
        draw = ImageDraw.Draw(im)
        draw.multiline_textbbox((10, 10), 'ABC\nAaaa', font, stroke_width=2)
        draw.text((10, 10), 'Test Text', font=font, fill='#000')