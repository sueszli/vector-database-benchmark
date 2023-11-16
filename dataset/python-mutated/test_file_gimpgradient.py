from PIL import GimpGradientFile, ImagePalette

def test_linear_pos_le_middle():
    if False:
        print('Hello World!')
    middle = 0.5
    pos = 0.25
    ret = GimpGradientFile.linear(middle, pos)
    assert ret == 0.25

def test_linear_pos_le_small_middle():
    if False:
        print('Hello World!')
    middle = 1e-11
    pos = 1e-12
    ret = GimpGradientFile.linear(middle, pos)
    assert ret == 0.0

def test_linear_pos_gt_middle():
    if False:
        while True:
            i = 10
    middle = 0.5
    pos = 0.75
    ret = GimpGradientFile.linear(middle, pos)
    assert ret == 0.75

def test_linear_pos_gt_small_middle():
    if False:
        for i in range(10):
            print('nop')
    middle = 1 - 1e-11
    pos = 1 - 1e-12
    ret = GimpGradientFile.linear(middle, pos)
    assert ret == 1.0

def test_curved():
    if False:
        for i in range(10):
            print('nop')
    middle = 0.5
    pos = 0.75
    ret = GimpGradientFile.curved(middle, pos)
    assert ret == 0.75

def test_sine():
    if False:
        return 10
    middle = 0.5
    pos = 0.75
    ret = GimpGradientFile.sine(middle, pos)
    assert ret == 0.8535533905932737

def test_sphere_increasing():
    if False:
        i = 10
        return i + 15
    middle = 0.5
    pos = 0.75
    ret = GimpGradientFile.sphere_increasing(middle, pos)
    assert round(abs(ret - 0.9682458365518543), 7) == 0

def test_sphere_decreasing():
    if False:
        while True:
            i = 10
    middle = 0.5
    pos = 0.75
    ret = GimpGradientFile.sphere_decreasing(middle, pos)
    assert ret == 0.3385621722338523

def test_load_via_imagepalette():
    if False:
        while True:
            i = 10
    test_file = 'Tests/images/gimp_gradient.ggr'
    palette = ImagePalette.load(test_file)
    assert len(palette[0]) == 1024
    assert palette[1] == 'RGBA'

def test_load_1_3_via_imagepalette():
    if False:
        return 10
    test_file = 'Tests/images/gimp_gradient_with_name.ggr'
    palette = ImagePalette.load(test_file)
    assert len(palette[0]) == 1024
    assert palette[1] == 'RGBA'