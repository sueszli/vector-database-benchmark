import lief
from utils import get_sample
lief.logging.set_level(lief.logging.LOGGING_LEVEL.INFO)

def test_art17():
    if False:
        return 10
    boot = lief.ART.parse(get_sample('ART/ART_017_AArch64_boot.art'))
    assert boot.header is not None

def test_art29():
    if False:
        print('Hello World!')
    boot = lief.ART.parse(get_sample('ART/ART_029_ARM_boot.art'))
    assert boot.header is not None

def test_art30():
    if False:
        for i in range(10):
            print('nop')
    boot = lief.ART.parse(get_sample('ART/ART_030_AArch64_boot.art'))
    assert boot.header is not None

def test_art44():
    if False:
        while True:
            i = 10
    boot = lief.ART.parse(get_sample('ART/ART_044_ARM_boot.art'))
    assert boot.header is not None

def test_art46():
    if False:
        for i in range(10):
            print('nop')
    boot = lief.ART.parse(get_sample('ART/ART_046_AArch64_boot.art'))
    assert boot.header is not None

def test_art56():
    if False:
        return 10
    boot = lief.ART.parse(get_sample('ART/ART_056_AArch64_boot.art'))
    assert boot.header is not None