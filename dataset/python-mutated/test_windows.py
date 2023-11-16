import os
import pytest
from tests.test_replay import replay
from .conftest import BOOTROM_FRAMES_UNTIL_LOGO
replay_file = 'tests/replays/default_rom.replay'

def test_headless(default_rom):
    if False:
        print('Hello World!')
    replay(default_rom, replay_file, 'headless', bootrom_file=None, padding_frames=BOOTROM_FRAMES_UNTIL_LOGO)

def test_dummy(default_rom):
    if False:
        while True:
            i = 10
    replay(default_rom, replay_file, 'dummy', bootrom_file=None, verify=False)

@pytest.mark.skipif(os.environ.get('TEST_NO_UI'), reason='Skipping test, as there is no UI')
def test_sdl2(default_rom):
    if False:
        return 10
    replay(default_rom, replay_file, 'SDL2', bootrom_file=None, padding_frames=BOOTROM_FRAMES_UNTIL_LOGO)

@pytest.mark.skipif(os.environ.get('TEST_NO_UI'), reason='Skipping test, as there is no UI')
def test_opengl(default_rom):
    if False:
        for i in range(10):
            print('nop')
    replay(default_rom, replay_file, 'OpenGL', bootrom_file=None, padding_frames=BOOTROM_FRAMES_UNTIL_LOGO)