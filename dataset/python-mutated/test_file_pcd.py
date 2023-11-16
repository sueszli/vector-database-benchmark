from PIL import Image

def test_load_raw():
    if False:
        return 10
    with Image.open('Tests/images/hopper.pcd') as im:
        im.load()