import unittest
from transformers import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING, TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING, PreTrainedTokenizerBase, is_vision_available
from transformers.pipelines import ImageClassificationPipeline, pipeline
from transformers.testing_utils import is_pipeline_test, nested_simplify, require_tf, require_torch, require_torch_or_tf, require_vision, slow
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
@require_torch_or_tf
@require_vision
class ImageClassificationPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING
    tf_model_mapping = TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING

    def get_test_pipeline(self, model, tokenizer, processor):
        if False:
            for i in range(10):
                print('nop')
        image_classifier = ImageClassificationPipeline(model=model, image_processor=processor, top_k=2)
        examples = [Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png'), 'http://images.cocodataset.org/val2017/000000039769.jpg']
        return (image_classifier, examples)

    def run_pipeline_test(self, image_classifier, examples):
        if False:
            return 10
        outputs = image_classifier('./tests/fixtures/tests_samples/COCO/000000039769.png')
        self.assertEqual(outputs, [{'score': ANY(float), 'label': ANY(str)}, {'score': ANY(float), 'label': ANY(str)}])
        import datasets
        dataset = datasets.load_dataset('hf-internal-testing/fixtures_image_utils', 'image', split='test')
        outputs = image_classifier([Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png'), 'http://images.cocodataset.org/val2017/000000039769.jpg', dataset[0]['file'], dataset[1]['file'], dataset[2]['file']])
        self.assertEqual(outputs, [[{'score': ANY(float), 'label': ANY(str)}, {'score': ANY(float), 'label': ANY(str)}], [{'score': ANY(float), 'label': ANY(str)}, {'score': ANY(float), 'label': ANY(str)}], [{'score': ANY(float), 'label': ANY(str)}, {'score': ANY(float), 'label': ANY(str)}], [{'score': ANY(float), 'label': ANY(str)}, {'score': ANY(float), 'label': ANY(str)}], [{'score': ANY(float), 'label': ANY(str)}, {'score': ANY(float), 'label': ANY(str)}]])

    @require_torch
    def test_small_model_pt(self):
        if False:
            print('Hello World!')
        small_model = 'hf-internal-testing/tiny-random-vit'
        image_classifier = pipeline('image-classification', model=small_model)
        outputs = image_classifier('http://images.cocodataset.org/val2017/000000039769.jpg')
        self.assertEqual(nested_simplify(outputs, decimals=4), [{'label': 'LABEL_1', 'score': 0.574}, {'label': 'LABEL_0', 'score': 0.426}])
        outputs = image_classifier(['http://images.cocodataset.org/val2017/000000039769.jpg', 'http://images.cocodataset.org/val2017/000000039769.jpg'], top_k=2)
        self.assertEqual(nested_simplify(outputs, decimals=4), [[{'label': 'LABEL_1', 'score': 0.574}, {'label': 'LABEL_0', 'score': 0.426}], [{'label': 'LABEL_1', 'score': 0.574}, {'label': 'LABEL_0', 'score': 0.426}]])

    @require_tf
    def test_small_model_tf(self):
        if False:
            while True:
                i = 10
        small_model = 'hf-internal-testing/tiny-random-vit'
        image_classifier = pipeline('image-classification', model=small_model, framework='tf')
        outputs = image_classifier('http://images.cocodataset.org/val2017/000000039769.jpg')
        self.assertEqual(nested_simplify(outputs, decimals=4), [{'label': 'LABEL_1', 'score': 0.574}, {'label': 'LABEL_0', 'score': 0.426}])
        outputs = image_classifier(['http://images.cocodataset.org/val2017/000000039769.jpg', 'http://images.cocodataset.org/val2017/000000039769.jpg'], top_k=2)
        self.assertEqual(nested_simplify(outputs, decimals=4), [[{'label': 'LABEL_1', 'score': 0.574}, {'label': 'LABEL_0', 'score': 0.426}], [{'label': 'LABEL_1', 'score': 0.574}, {'label': 'LABEL_0', 'score': 0.426}]])

    def test_custom_tokenizer(self):
        if False:
            return 10
        tokenizer = PreTrainedTokenizerBase()
        image_classifier = pipeline('image-classification', model='hf-internal-testing/tiny-random-vit', tokenizer=tokenizer)
        self.assertIs(image_classifier.tokenizer, tokenizer)

    @slow
    @require_torch
    def test_perceiver(self):
        if False:
            i = 10
            return i + 15
        image_classifier = pipeline('image-classification', model='deepmind/vision-perceiver-conv')
        outputs = image_classifier('http://images.cocodataset.org/val2017/000000039769.jpg')
        self.assertEqual(nested_simplify(outputs, decimals=4), [{'score': 0.4385, 'label': 'tabby, tabby cat'}, {'score': 0.321, 'label': 'tiger cat'}, {'score': 0.0502, 'label': 'Egyptian cat'}, {'score': 0.0137, 'label': 'crib, cot'}, {'score': 0.007, 'label': 'radiator'}])
        image_classifier = pipeline('image-classification', model='deepmind/vision-perceiver-fourier')
        outputs = image_classifier('http://images.cocodataset.org/val2017/000000039769.jpg')
        self.assertEqual(nested_simplify(outputs, decimals=4), [{'score': 0.5658, 'label': 'tabby, tabby cat'}, {'score': 0.1309, 'label': 'tiger cat'}, {'score': 0.0722, 'label': 'Egyptian cat'}, {'score': 0.0707, 'label': 'remote control, remote'}, {'score': 0.0082, 'label': 'computer keyboard, keypad'}])
        image_classifier = pipeline('image-classification', model='deepmind/vision-perceiver-learned')
        outputs = image_classifier('http://images.cocodataset.org/val2017/000000039769.jpg')
        self.assertEqual(nested_simplify(outputs, decimals=4), [{'score': 0.3022, 'label': 'tabby, tabby cat'}, {'score': 0.2362, 'label': 'Egyptian cat'}, {'score': 0.1856, 'label': 'tiger cat'}, {'score': 0.0324, 'label': 'remote control, remote'}, {'score': 0.0096, 'label': 'quilt, comforter, comfort, puff'}])