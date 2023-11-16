from rich.color_triplet import ColorTriplet

def test_hex():
    if False:
        i = 10
        return i + 15
    assert ColorTriplet(255, 255, 255).hex == '#ffffff'
    assert ColorTriplet(0, 255, 0).hex == '#00ff00'

def test_rgb():
    if False:
        print('Hello World!')
    assert ColorTriplet(255, 255, 255).rgb == 'rgb(255,255,255)'
    assert ColorTriplet(0, 255, 0).rgb == 'rgb(0,255,0)'

def test_normalized():
    if False:
        i = 10
        return i + 15
    assert ColorTriplet(255, 255, 255).normalized == (1.0, 1.0, 1.0)
    assert ColorTriplet(0, 255, 0).normalized == (0.0, 1.0, 0.0)