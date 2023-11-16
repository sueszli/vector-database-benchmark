import pytest
from packaging.version import parse as parse_version
from PIL import Image, features
from .helper import assert_image_equal, assert_image_similar, is_big_endian, skip_unless_feature
pytestmark = [skip_unless_feature('webp'), skip_unless_feature('webp_anim')]

def test_n_frames():
    if False:
        return 10
    'Ensure that WebP format sets n_frames and is_animated attributes correctly.'
    with Image.open('Tests/images/hopper.webp') as im:
        assert im.n_frames == 1
        assert not im.is_animated
    with Image.open('Tests/images/iss634.webp') as im:
        assert im.n_frames == 42
        assert im.is_animated

def test_write_animation_L(tmp_path):
    if False:
        print('Hello World!')
    "\n    Convert an animated GIF to animated WebP, then compare the frame count, and first\n    and last frames to ensure they're visually similar.\n    "
    with Image.open('Tests/images/iss634.gif') as orig:
        assert orig.n_frames > 1
        temp_file = str(tmp_path / 'temp.webp')
        orig.save(temp_file, save_all=True)
        with Image.open(temp_file) as im:
            assert im.n_frames == orig.n_frames
            orig.load()
            im.load()
            assert_image_similar(im, orig.convert('RGBA'), 32.9)
            if is_big_endian():
                webp = parse_version(features.version_module('webp'))
                if webp < parse_version('1.2.2'):
                    pytest.skip('Fails with libwebp earlier than 1.2.2')
            orig.seek(orig.n_frames - 1)
            im.seek(im.n_frames - 1)
            orig.load()
            im.load()
            assert_image_similar(im, orig.convert('RGBA'), 32.9)

def test_write_animation_RGB(tmp_path):
    if False:
        return 10
    '\n    Write an animated WebP from RGB frames, and ensure the frames\n    are visually similar to the originals.\n    '

    def check(temp_file):
        if False:
            for i in range(10):
                print('nop')
        with Image.open(temp_file) as im:
            assert im.n_frames == 2
            im.load()
            assert_image_equal(im, frame1.convert('RGBA'))
            if is_big_endian():
                webp = parse_version(features.version_module('webp'))
                if webp < parse_version('1.2.2'):
                    pytest.skip('Fails with libwebp earlier than 1.2.2')
            im.seek(1)
            im.load()
            assert_image_equal(im, frame2.convert('RGBA'))
    with Image.open('Tests/images/anim_frame1.webp') as frame1:
        with Image.open('Tests/images/anim_frame2.webp') as frame2:
            temp_file1 = str(tmp_path / 'temp.webp')
            frame1.copy().save(temp_file1, save_all=True, append_images=[frame2], lossless=True)
            check(temp_file1)

            def im_generator(ims):
                if False:
                    print('Hello World!')
                yield from ims
            temp_file2 = str(tmp_path / 'temp_generator.webp')
            frame1.copy().save(temp_file2, save_all=True, append_images=im_generator([frame2]), lossless=True)
            check(temp_file2)

def test_timestamp_and_duration(tmp_path):
    if False:
        i = 10
        return i + 15
    '\n    Try passing a list of durations, and make sure the encoded\n    timestamps and durations are correct.\n    '
    durations = [0, 10, 20, 30, 40]
    temp_file = str(tmp_path / 'temp.webp')
    with Image.open('Tests/images/anim_frame1.webp') as frame1:
        with Image.open('Tests/images/anim_frame2.webp') as frame2:
            frame1.save(temp_file, save_all=True, append_images=[frame2, frame1, frame2, frame1], duration=durations)
    with Image.open(temp_file) as im:
        assert im.n_frames == 5
        assert im.is_animated
        ts = 0
        for frame in range(im.n_frames):
            im.seek(frame)
            im.load()
            assert im.info['duration'] == durations[frame]
            assert im.info['timestamp'] == ts
            ts += durations[frame]

def test_float_duration(tmp_path):
    if False:
        print('Hello World!')
    temp_file = str(tmp_path / 'temp.webp')
    with Image.open('Tests/images/iss634.apng') as im:
        assert im.info['duration'] == 70.0
        im.save(temp_file, save_all=True)
    with Image.open(temp_file) as reloaded:
        reloaded.load()
        assert reloaded.info['duration'] == 70

def test_seeking(tmp_path):
    if False:
        print('Hello World!')
    '\n    Create an animated WebP file, and then try seeking through frames in reverse-order,\n    verifying the timestamps and durations are correct.\n    '
    dur = 33
    temp_file = str(tmp_path / 'temp.webp')
    with Image.open('Tests/images/anim_frame1.webp') as frame1:
        with Image.open('Tests/images/anim_frame2.webp') as frame2:
            frame1.save(temp_file, save_all=True, append_images=[frame2, frame1, frame2, frame1], duration=dur)
    with Image.open(temp_file) as im:
        assert im.n_frames == 5
        assert im.is_animated
        ts = dur * (im.n_frames - 1)
        for frame in reversed(range(im.n_frames)):
            im.seek(frame)
            im.load()
            assert im.info['duration'] == dur
            assert im.info['timestamp'] == ts
            ts -= dur

def test_seek_errors():
    if False:
        for i in range(10):
            print('nop')
    with Image.open('Tests/images/iss634.webp') as im:
        with pytest.raises(EOFError):
            im.seek(-1)
        with pytest.raises(EOFError):
            im.seek(42)