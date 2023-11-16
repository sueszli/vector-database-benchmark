from PIL import Image, ImageDraw, ImageFont
from .helper import PillowLeakTestCase, skip_unless_feature

class TestTTypeFontLeak(PillowLeakTestCase):
    iterations = 10
    mem_limit = 4096

    def _test_font(self, font):
        if False:
            i = 10
            return i + 15
        im = Image.new('RGB', (255, 255), 'white')
        draw = ImageDraw.ImageDraw(im)
        self._test_leak(lambda : draw.text((0, 0), 'some text ' * 1024, font=font, fill='black'))

    @skip_unless_feature('freetype2')
    def test_leak(self):
        if False:
            return 10
        ttype = ImageFont.truetype('Tests/fonts/FreeMono.ttf', 20)
        self._test_font(ttype)

class TestDefaultFontLeak(TestTTypeFontLeak):
    iterations = 100
    mem_limit = 1024

    def test_leak(self):
        if False:
            return 10
        default_font = ImageFont.load_default()
        self._test_font(default_font)