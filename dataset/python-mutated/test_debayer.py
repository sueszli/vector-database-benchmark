import itertools
import os
import unittest
import numpy as np
from nvidia.dali import pipeline_def, fn, types
from test_utils import get_dali_extra_path
from nose_utils import assert_raises
from nose2.tools import params
from debayer_test_utils import bayer_patterns, blue_position, blue_position2pattern, rgb2bayer, rgb2bayer_seq, debayer_bilinear_npp_pattern, debayer_bilinear_npp_pattern_seq
data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')
vid_dir = os.path.join(data_root, 'db', 'video', 'sintel', 'video_files')
vid_files = ['sintel_trailer-720p_3.mp4']

def read_imgs(num_imgs, dtype, seed):
    if False:
        return 10

    @pipeline_def
    def pipeline():
        if False:
            for i in range(10):
                print('nop')
        (input, _) = fn.readers.file(file_root=images_dir, random_shuffle=True, seed=seed)
        return fn.decoders.image(input, device='cpu', output_type=types.RGB)
    pipe = pipeline(batch_size=num_imgs, device_id=0, num_threads=4)
    pipe.build()
    (batch,) = pipe.run()
    return [np.array(img, dtype=dtype) for img in batch]

def read_video(num_sequences, num_frames, height, width, seed=42):
    if False:
        while True:
            i = 10
    roi_start = (90, 0)
    roi_end = (630, 1280)
    vid_filenames = [os.path.join(vid_dir, vid_file) for vid_file in vid_files]

    @pipeline_def
    def pipeline():
        if False:
            return 10
        video = fn.readers.video_resize(filenames=vid_filenames, name='video reader', sequence_length=num_frames, file_list_include_preceding_frame=True, device='gpu', roi_start=roi_start, roi_end=roi_end, seed=seed, resize_x=width, resize_y=height)
        return video
    pipe = pipeline(batch_size=num_sequences, device_id=0, num_threads=4)
    pipe.build()
    (batch,) = pipe.run()
    return [np.array(seq) for seq in batch.as_cpu()]

def prepare_test_imgs(num_samples, dtype):
    if False:
        print('Hello World!')
    assert dtype in (np.uint8, np.uint16)
    rng = np.random.default_rng(seed=101)
    imgs = read_imgs(num_samples, dtype, seed=42 if dtype == np.uint8 else 13)
    if dtype == np.uint16:
        imgs = [np.uint16(img) * 256 + np.uint16(rng.uniform(0, 256, size=img.shape)) for img in imgs]
    bayered_imgs = {pattern: [rgb2bayer(img, pattern) for img in imgs] for pattern in bayer_patterns}
    npp_baseline = {pattern: [debayer_bilinear_npp_pattern(img, pattern) for img in imgs] for (pattern, imgs) in bayered_imgs.items()}
    return (bayered_imgs, npp_baseline)

class DebayerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls.num_samples = 7
        (cls.bayered_imgs, cls.npp_baseline) = prepare_test_imgs(cls.num_samples, dtype=np.uint8)
        (cls.bayered_imgs16t, cls.npp_baseline16t) = prepare_test_imgs(cls.num_samples, dtype=np.uint16)

    @classmethod
    def get_test_data(cls, dtype):
        if False:
            print('Hello World!')
        assert dtype in (np.uint8, np.uint16)
        if dtype == np.uint8:
            return (cls.bayered_imgs, cls.npp_baseline)
        else:
            return (cls.bayered_imgs16t, cls.npp_baseline16t)

    @params(*enumerate(itertools.product([1, 64], bayer_patterns)))
    def test_debayer_fixed_pattern(self, i, args):
        if False:
            i = 10
            return i + 15
        (batch_size, pattern) = args
        num_iterations = 3
        test_hwc_single_channel_input = i % 2 == 1
        (bayered_imgs, npp_baseline) = self.get_test_data(np.uint8)

        def source(sample_info):
            if False:
                for i in range(10):
                    print('nop')
            idx = sample_info.idx_in_epoch % self.num_samples
            img = bayered_imgs[pattern][idx]
            assert len(img.shape) == 2
            if test_hwc_single_channel_input:
                (h, w) = img.shape
                img = img.reshape(h, w, 1)
            return (img, np.array(idx, dtype=np.int32))

        @pipeline_def
        def debayer_pipeline():
            if False:
                print('Hello World!')
            (bayer_imgs, idxs) = fn.external_source(source=source, batch=False, num_outputs=2)
            debayered_imgs = fn.experimental.debayer(bayer_imgs.gpu(), blue_position=blue_position(pattern))
            return (debayered_imgs, idxs)
        pipe = debayer_pipeline(batch_size=batch_size, device_id=0, num_threads=4)
        pipe.build()
        out_batches = []
        for _ in range(num_iterations):
            (debayered_imgs_dev, idxs) = pipe.run()
            assert debayered_imgs_dev.layout() == 'HWC'
            out_batches.append(([np.array(img) for img in debayered_imgs_dev.as_cpu()], [np.array(idx) for idx in idxs]))
        for (debayered_imgs, idxs) in out_batches:
            assert len(debayered_imgs) == len(idxs)
            for (img_debayered, idx) in zip(debayered_imgs, idxs):
                baseline = npp_baseline[pattern][idx]
                assert np.all(img_debayered == baseline)

    @params(*itertools.product([1, 11, 184], [np.uint8, np.uint16]))
    def test_debayer_per_sample_pattern(self, batch_size, dtype):
        if False:
            while True:
                i = 10
        num_iterations = 3
        num_patterns = len(bayer_patterns)
        rng = np.random.default_rng(seed=42 + batch_size)
        (bayered_imgs, npp_baseline) = self.get_test_data(dtype)

        def source(sample_info):
            if False:
                return 10
            idx = sample_info.idx_in_epoch % self.num_samples
            pattern_idx = np.int32(rng.uniform(0, num_patterns))
            pattern = bayer_patterns[pattern_idx]
            return (bayered_imgs[pattern][idx], np.array(blue_position(pattern), dtype=np.int32), np.array(idx, dtype=np.int32))

        @pipeline_def
        def debayer_pipeline():
            if False:
                return 10
            (bayer_imgs, blue_poses, idxs) = fn.external_source(source=source, batch=False, num_outputs=3)
            debayered_imgs = fn.experimental.debayer(bayer_imgs.gpu(), blue_position=blue_poses)
            return (debayered_imgs, blue_poses, idxs)
        pipe = debayer_pipeline(batch_size=batch_size, device_id=0, num_threads=4)
        pipe.build()
        out_batches = []
        for _ in range(num_iterations):
            (debayered_imgs_dev, blue_poses, idxs) = pipe.run()
            assert debayered_imgs_dev.layout() == 'HWC'
            out_batches.append(([np.array(img) for img in debayered_imgs_dev.as_cpu()], [blue_position2pattern(np.array(blue_pos)) for blue_pos in blue_poses], [np.array(idx) for idx in idxs]))
        for (debayered_imgs, patterns, idxs) in out_batches:
            assert len(debayered_imgs) == len(patterns) == len(idxs)
            for (img_debayered, pattern, idx) in zip(debayered_imgs, patterns, idxs):
                baseline = npp_baseline[pattern][idx]
                assert np.all(img_debayered == baseline)

class DebayerVideoTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        rng = np.random.default_rng(seed=3)
        (num_smaller, num_bigger) = (4, 3)
        cls.num_samples = num_smaller + num_bigger
        smaller = read_video(num_smaller, 60, 108, 192)
        bigger = read_video(num_bigger, 32, 216, 384)
        video = smaller + bigger
        rng.shuffle(video)
        patterns = [rng.choice(bayer_patterns, len(vid)) for vid in video]
        cls.blue_poses = [np.array([blue_position(pattern) for pattern in sample_patterns], dtype=np.int32) for sample_patterns in patterns]
        cls.bayered_vid = [rgb2bayer_seq(vid, vid_patterns) for (vid, vid_patterns) in zip(video, patterns)]
        cls.npp_baseline = [debayer_bilinear_npp_pattern_seq(vid, vid_patterns) for (vid, vid_patterns) in zip(cls.bayered_vid, patterns)]

    def test_debayer_vid_per_frame_pattern(self):
        if False:
            print('Hello World!')
        num_iterations = 2
        batch_size = (self.num_samples + 1) // 2

        def source(sample_info):
            if False:
                return 10
            idx = sample_info.idx_in_epoch % self.num_samples
            vid = self.bayered_vid[idx]
            return (vid, self.blue_poses[idx], np.array(idx, dtype=np.int32))

        @pipeline_def
        def debayer_pipeline():
            if False:
                while True:
                    i = 10
            (bayered_vid, blue_positions, idxs) = fn.external_source(source=source, batch=False, num_outputs=3, layout=['FHW', None, None])
            debayered_vid = fn.experimental.debayer(bayered_vid.gpu(), blue_position=fn.per_frame(blue_positions))
            return (debayered_vid, idxs)
        pipe = debayer_pipeline(batch_size=batch_size, device_id=0, num_threads=4)
        pipe.build()
        out_batches = []
        for _ in range(num_iterations):
            (debayered_dev, idxs) = pipe.run()
            assert debayered_dev.layout() == 'FHWC'
            out_batches.append(([np.array(vid) for vid in debayered_dev.as_cpu()], [np.array(idx) for idx in idxs]))
        for (debayered_videos, idxs) in out_batches:
            assert len(debayered_videos) == len(idxs)
            for (vid_debayered, idx) in zip(debayered_videos, idxs):
                baseline = self.npp_baseline[idx]
                assert np.all(vid_debayered == baseline)

def source_full_array(shape, dtype):
    if False:
        print('Hello World!')

    def source(sample_info):
        if False:
            for i in range(10):
                print('nop')
        return np.full(shape, sample_info.idx_in_epoch, dtype=dtype)
    return source

def _test_shape_pipeline(shape, dtype):
    if False:
        i = 10
        return i + 15

    @pipeline_def
    def pipeline():
        if False:
            for i in range(10):
                print('nop')
        bayer_imgs = fn.external_source(source_full_array(shape, dtype), batch=False)
        return fn.experimental.debayer(bayer_imgs.gpu(), blue_position=[0, 0])
    pipe = pipeline(batch_size=8, num_threads=4, device_id=0)
    pipe.build()
    pipe.run()

def test_odd_size_error():
    if False:
        for i in range(10):
            print('nop')
    with assert_raises(RuntimeError, glob='The height and width of the image to debayer must be even'):
        _test_shape_pipeline((20, 15), np.uint8)

def test_too_many_channels():
    if False:
        return 10
    with assert_raises(RuntimeError, glob=' The debayer operator expects grayscale (i.e. single channel) images'):
        _test_shape_pipeline((20, 40, 2), np.uint8)
    with assert_raises(RuntimeError, glob=' The debayer operator expects grayscale (i.e. single channel) images'):
        _test_shape_pipeline((20, 40, 2, 2), np.uint8)

def test_wrong_sample_dim():
    if False:
        return 10
    with assert_raises(RuntimeError, glob='The number of dimensions 5 does not match any of the allowed'):
        _test_shape_pipeline((1, 1, 1, 1, 1), np.uint8)

def test_no_blue_position_specified():
    if False:
        i = 10
        return i + 15
    with assert_raises(RuntimeError, glob='Not all required arguments were specified'):

        @pipeline_def
        def pipeline():
            if False:
                return 10
            bayer_imgs = fn.external_source(source_full_array((20, 20), np.uint8), batch=False)
            return fn.experimental.debayer(bayer_imgs.gpu())
        pipe = pipeline(batch_size=8, num_threads=4, device_id=0)
        pipe.build()
        pipe.run()

@params(((2, 2),), ((1, 2),), ((-1, 0),))
def test_blue_position_outside_of_2x2_tile(blue_position):
    if False:
        return 10
    with assert_raises(RuntimeError, glob='The `blue_position` position must lie within 2x2 tile'):

        @pipeline_def
        def pipeline():
            if False:
                i = 10
                return i + 15
            bayer_imgs = fn.external_source(source_full_array((20, 20), np.uint8), batch=False)
            return fn.experimental.debayer(bayer_imgs.gpu(), blue_position=blue_position)
        pipe = pipeline(batch_size=8, num_threads=4, device_id=0)
        pipe.build()
        pipe.run()