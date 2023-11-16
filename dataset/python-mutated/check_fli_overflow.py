from PIL import Image
TEST_FILE = 'Tests/images/fli_overflow.fli'

def test_fli_overflow():
    if False:
        i = 10
        return i + 15
    with Image.open(TEST_FILE) as im:
        im.load()