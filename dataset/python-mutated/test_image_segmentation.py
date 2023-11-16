import unittest
from pathlib import Path
from transformers import is_vision_available, load_tool
from transformers.testing_utils import get_tests_dir
from .test_tools_common import ToolTesterMixin
if is_vision_available():
    from PIL import Image

class ImageSegmentationToolTester(unittest.TestCase, ToolTesterMixin):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tool = load_tool('image-segmentation')
        self.tool.setup()
        self.remote_tool = load_tool('image-segmentation', remote=True)

    def test_exact_match_arg(self):
        if False:
            return 10
        image = Image.open(Path(get_tests_dir('fixtures/tests_samples/COCO')) / '000000039769.png')
        result = self.tool(image, 'cat')
        self.assertTrue(isinstance(result, Image.Image))

    def test_exact_match_arg_remote(self):
        if False:
            i = 10
            return i + 15
        image = Image.open(Path(get_tests_dir('fixtures/tests_samples/COCO')) / '000000039769.png')
        result = self.remote_tool(image, 'cat')
        self.assertTrue(isinstance(result, Image.Image))

    def test_exact_match_kwarg(self):
        if False:
            print('Hello World!')
        image = Image.open(Path(get_tests_dir('fixtures/tests_samples/COCO')) / '000000039769.png')
        result = self.tool(image=image, label='cat')
        self.assertTrue(isinstance(result, Image.Image))

    def test_exact_match_kwarg_remote(self):
        if False:
            for i in range(10):
                print('nop')
        image = Image.open(Path(get_tests_dir('fixtures/tests_samples/COCO')) / '000000039769.png')
        result = self.remote_tool(image=image, label='cat')
        self.assertTrue(isinstance(result, Image.Image))