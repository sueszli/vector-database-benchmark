import unittest
from transformers import is_vision_available
from transformers.pipelines import pipeline
from transformers.testing_utils import is_pipeline_test, nested_simplify, require_tf, require_torch, require_vision, slow
from .test_pipelines_common import ANY
if is_vision_available():
    from PIL import Image
else:

    class Image:

        @staticmethod
        def open(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            pass

@is_pipeline_test
@require_vision
class ZeroShotImageClassificationPipelineTests(unittest.TestCase):

    @require_torch
    def test_small_model_pt(self):
        if False:
            print('Hello World!')
        image_classifier = pipeline(model='hf-internal-testing/tiny-random-clip-zero-shot-image-classification')
        image = Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png')
        output = image_classifier(image, candidate_labels=['a', 'b', 'c'])
        self.assertIn(nested_simplify(output), [[{'score': 0.333, 'label': 'a'}, {'score': 0.333, 'label': 'b'}, {'score': 0.333, 'label': 'c'}], [{'score': 0.333, 'label': 'a'}, {'score': 0.333, 'label': 'c'}, {'score': 0.333, 'label': 'b'}], [{'score': 0.333, 'label': 'b'}, {'score': 0.333, 'label': 'a'}, {'score': 0.333, 'label': 'c'}]])
        output = image_classifier([image] * 5, candidate_labels=['A', 'B', 'C'], batch_size=2)
        self.assertEqual(nested_simplify(output), [[{'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}], [{'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}], [{'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}], [{'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}], [{'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}]])

    @require_tf
    def test_small_model_tf(self):
        if False:
            for i in range(10):
                print('nop')
        image_classifier = pipeline(model='hf-internal-testing/tiny-random-clip-zero-shot-image-classification', framework='tf')
        image = Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png')
        output = image_classifier(image, candidate_labels=['a', 'b', 'c'])
        self.assertEqual(nested_simplify(output), [{'score': 0.333, 'label': 'a'}, {'score': 0.333, 'label': 'b'}, {'score': 0.333, 'label': 'c'}])
        output = image_classifier([image] * 5, candidate_labels=['A', 'B', 'C'], batch_size=2)
        self.assertEqual(nested_simplify(output), [[{'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}], [{'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}], [{'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}], [{'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}], [{'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}, {'score': 0.333, 'label': ANY(str)}]])

    @slow
    @require_torch
    def test_large_model_pt(self):
        if False:
            return 10
        image_classifier = pipeline(task='zero-shot-image-classification', model='openai/clip-vit-base-patch32')
        image = Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png')
        output = image_classifier(image, candidate_labels=['cat', 'plane', 'remote'])
        self.assertEqual(nested_simplify(output), [{'score': 0.511, 'label': 'remote'}, {'score': 0.485, 'label': 'cat'}, {'score': 0.004, 'label': 'plane'}])
        output = image_classifier([image] * 5, candidate_labels=['cat', 'plane', 'remote'], batch_size=2)
        self.assertEqual(nested_simplify(output), [[{'score': 0.511, 'label': 'remote'}, {'score': 0.485, 'label': 'cat'}, {'score': 0.004, 'label': 'plane'}]] * 5)

    @slow
    @require_tf
    def test_large_model_tf(self):
        if False:
            while True:
                i = 10
        image_classifier = pipeline(task='zero-shot-image-classification', model='openai/clip-vit-base-patch32', framework='tf')
        image = Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png')
        output = image_classifier(image, candidate_labels=['cat', 'plane', 'remote'])
        self.assertEqual(nested_simplify(output), [{'score': 0.511, 'label': 'remote'}, {'score': 0.485, 'label': 'cat'}, {'score': 0.004, 'label': 'plane'}])
        output = image_classifier([image] * 5, candidate_labels=['cat', 'plane', 'remote'], batch_size=2)
        self.assertEqual(nested_simplify(output), [[{'score': 0.511, 'label': 'remote'}, {'score': 0.485, 'label': 'cat'}, {'score': 0.004, 'label': 'plane'}]] * 5)