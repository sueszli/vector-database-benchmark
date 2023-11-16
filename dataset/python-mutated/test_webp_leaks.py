from io import BytesIO
from PIL import Image
from .helper import PillowLeakTestCase, skip_unless_feature
test_file = 'Tests/images/hopper.webp'

@skip_unless_feature('webp')
class TestWebPLeaks(PillowLeakTestCase):
    mem_limit = 3 * 1024
    iterations = 100

    def test_leak_load(self):
        if False:
            while True:
                i = 10
        with open(test_file, 'rb') as f:
            im_data = f.read()

        def core():
            if False:
                for i in range(10):
                    print('nop')
            with Image.open(BytesIO(im_data)) as im:
                im.load()
        self._test_leak(core)