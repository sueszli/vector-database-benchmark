import shutil
import tempfile
import unittest
import numpy as np
import pytest
from transformers import is_speech_available, is_vision_available
from transformers.testing_utils import require_torch
if is_vision_available():
    from transformers import TvltImageProcessor
if is_speech_available():
    from transformers import TvltFeatureExtractor
from transformers import TvltProcessor

@require_torch
class TvltProcessorTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.checkpoint = 'ZinengTang/tvlt-base'
        self.tmpdirname = tempfile.mkdtemp()

    def get_image_processor(self, **kwargs):
        if False:
            while True:
                i = 10
        return TvltImageProcessor.from_pretrained(self.checkpoint, **kwargs)

    def get_feature_extractor(self, **kwargs):
        if False:
            while True:
                i = 10
        return TvltFeatureExtractor.from_pretrained(self.checkpoint, **kwargs)

    def tearDown(self):
        if False:
            return 10
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        if False:
            print('Hello World!')
        image_processor = self.get_image_processor()
        feature_extractor = self.get_feature_extractor()
        processor = TvltProcessor(image_processor=image_processor, feature_extractor=feature_extractor)
        processor.save_pretrained(self.tmpdirname)
        processor = TvltProcessor.from_pretrained(self.tmpdirname)
        self.assertIsInstance(processor.feature_extractor, TvltFeatureExtractor)
        self.assertIsInstance(processor.image_processor, TvltImageProcessor)

    def test_feature_extractor(self):
        if False:
            return 10
        image_processor = self.get_image_processor()
        feature_extractor = self.get_feature_extractor()
        processor = TvltProcessor(image_processor=image_processor, feature_extractor=feature_extractor)
        audio = np.ones([12000])
        audio_dict = feature_extractor(audio, return_tensors='np')
        input_processor = processor(audio=audio, return_tensors='np')
        for key in audio_dict.keys():
            self.assertAlmostEqual(audio_dict[key].sum(), input_processor[key].sum(), delta=0.01)

    def test_image_processor(self):
        if False:
            i = 10
            return i + 15
        image_processor = self.get_image_processor()
        feature_extractor = self.get_feature_extractor()
        processor = TvltProcessor(image_processor=image_processor, feature_extractor=feature_extractor)
        images = np.ones([3, 224, 224])
        image_dict = image_processor(images, return_tensors='np')
        input_processor = processor(images=images, return_tensors='np')
        for key in image_dict.keys():
            self.assertAlmostEqual(image_dict[key].sum(), input_processor[key].sum(), delta=0.01)

    def test_processor(self):
        if False:
            while True:
                i = 10
        image_processor = self.get_image_processor()
        feature_extractor = self.get_feature_extractor()
        processor = TvltProcessor(image_processor=image_processor, feature_extractor=feature_extractor)
        audio = np.ones([12000])
        images = np.ones([3, 224, 224])
        inputs = processor(audio=audio, images=images)
        self.assertListEqual(list(inputs.keys()), ['audio_values', 'audio_mask', 'pixel_values', 'pixel_mask'])
        with pytest.raises(ValueError):
            processor()

    def test_model_input_names(self):
        if False:
            while True:
                i = 10
        image_processor = self.get_image_processor()
        feature_extractor = self.get_feature_extractor()
        processor = TvltProcessor(image_processor=image_processor, feature_extractor=feature_extractor)
        self.assertListEqual(processor.model_input_names, image_processor.model_input_names + feature_extractor.model_input_names, msg='`processor` and `image_processor`+`feature_extractor` model input names do not match')