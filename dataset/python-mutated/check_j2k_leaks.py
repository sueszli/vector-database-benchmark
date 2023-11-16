from io import BytesIO
import pytest
from PIL import Image
from .helper import is_win32, skip_unless_feature
mem_limit = 1024 * 1048576
stack_size = 8 * 1048576
iterations = int(mem_limit / stack_size * 2)
test_file = 'Tests/images/rgb_trns_ycbc.jp2'
pytestmark = [pytest.mark.skipif(is_win32(), reason='requires Unix or macOS'), skip_unless_feature('jpg_2000')]

def test_leak_load():
    if False:
        i = 10
        return i + 15
    from resource import RLIMIT_AS, RLIMIT_STACK, setrlimit
    setrlimit(RLIMIT_STACK, (stack_size, stack_size))
    setrlimit(RLIMIT_AS, (mem_limit, mem_limit))
    for _ in range(iterations):
        with Image.open(test_file) as im:
            im.load()

def test_leak_save():
    if False:
        while True:
            i = 10
    from resource import RLIMIT_AS, RLIMIT_STACK, setrlimit
    setrlimit(RLIMIT_STACK, (stack_size, stack_size))
    setrlimit(RLIMIT_AS, (mem_limit, mem_limit))
    for _ in range(iterations):
        with Image.open(test_file) as im:
            im.load()
            test_output = BytesIO()
            im.save(test_output, 'JPEG2000')
            test_output.seek(0)
            test_output.read()