import unittest
from transformers import MODEL_FOR_OBJECT_DETECTION_MAPPING, AutoFeatureExtractor, AutoModelForObjectDetection, ObjectDetectionPipeline, is_vision_available, pipeline
from transformers.testing_utils import is_pipeline_test, nested_simplify, require_pytesseract, require_tf, require_timm, require_torch, require_vision, slow
from .test_pipelines_common import ANY
if is_vision_available():
    from PIL import Image
else:

    class Image:

        @staticmethod
        def open(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            pass

@is_pipeline_test
@require_vision
@require_timm
@require_torch
class ObjectDetectionPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_OBJECT_DETECTION_MAPPING

    def get_test_pipeline(self, model, tokenizer, processor):
        if False:
            while True:
                i = 10
        object_detector = ObjectDetectionPipeline(model=model, image_processor=processor)
        return (object_detector, ['./tests/fixtures/tests_samples/COCO/000000039769.png'])

    def run_pipeline_test(self, object_detector, examples):
        if False:
            while True:
                i = 10
        outputs = object_detector('./tests/fixtures/tests_samples/COCO/000000039769.png', threshold=0.0)
        self.assertGreater(len(outputs), 0)
        for detected_object in outputs:
            self.assertEqual(detected_object, {'score': ANY(float), 'label': ANY(str), 'box': {'xmin': ANY(int), 'ymin': ANY(int), 'xmax': ANY(int), 'ymax': ANY(int)}})
        import datasets
        dataset = datasets.load_dataset('hf-internal-testing/fixtures_image_utils', 'image', split='test')
        batch = [Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png'), 'http://images.cocodataset.org/val2017/000000039769.jpg', dataset[0]['file'], dataset[1]['file'], dataset[2]['file']]
        batch_outputs = object_detector(batch, threshold=0.0)
        self.assertEqual(len(batch), len(batch_outputs))
        for outputs in batch_outputs:
            self.assertGreater(len(outputs), 0)
            for detected_object in outputs:
                self.assertEqual(detected_object, {'score': ANY(float), 'label': ANY(str), 'box': {'xmin': ANY(int), 'ymin': ANY(int), 'xmax': ANY(int), 'ymax': ANY(int)}})

    @require_tf
    @unittest.skip('Object detection not implemented in TF')
    def test_small_model_tf(self):
        if False:
            print('Hello World!')
        pass

    @require_torch
    def test_small_model_pt(self):
        if False:
            for i in range(10):
                print('nop')
        model_id = 'hf-internal-testing/tiny-detr-mobilenetsv3'
        model = AutoModelForObjectDetection.from_pretrained(model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        object_detector = ObjectDetectionPipeline(model=model, feature_extractor=feature_extractor)
        outputs = object_detector('http://images.cocodataset.org/val2017/000000039769.jpg', threshold=0.0)
        self.assertEqual(nested_simplify(outputs, decimals=4), [{'score': 0.3376, 'label': 'LABEL_0', 'box': {'xmin': 159, 'ymin': 120, 'xmax': 480, 'ymax': 359}}, {'score': 0.3376, 'label': 'LABEL_0', 'box': {'xmin': 159, 'ymin': 120, 'xmax': 480, 'ymax': 359}}])
        outputs = object_detector(['http://images.cocodataset.org/val2017/000000039769.jpg', 'http://images.cocodataset.org/val2017/000000039769.jpg'], threshold=0.0)
        self.assertEqual(nested_simplify(outputs, decimals=4), [[{'score': 0.3376, 'label': 'LABEL_0', 'box': {'xmin': 159, 'ymin': 120, 'xmax': 480, 'ymax': 359}}, {'score': 0.3376, 'label': 'LABEL_0', 'box': {'xmin': 159, 'ymin': 120, 'xmax': 480, 'ymax': 359}}], [{'score': 0.3376, 'label': 'LABEL_0', 'box': {'xmin': 159, 'ymin': 120, 'xmax': 480, 'ymax': 359}}, {'score': 0.3376, 'label': 'LABEL_0', 'box': {'xmin': 159, 'ymin': 120, 'xmax': 480, 'ymax': 359}}]])

    @require_torch
    @slow
    def test_large_model_pt(self):
        if False:
            while True:
                i = 10
        model_id = 'facebook/detr-resnet-50'
        model = AutoModelForObjectDetection.from_pretrained(model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        object_detector = ObjectDetectionPipeline(model=model, feature_extractor=feature_extractor)
        outputs = object_detector('http://images.cocodataset.org/val2017/000000039769.jpg')
        self.assertEqual(nested_simplify(outputs, decimals=4), [{'score': 0.9982, 'label': 'remote', 'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}}, {'score': 0.996, 'label': 'remote', 'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}}, {'score': 0.9955, 'label': 'couch', 'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}}, {'score': 0.9988, 'label': 'cat', 'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}}, {'score': 0.9987, 'label': 'cat', 'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}])
        outputs = object_detector(['http://images.cocodataset.org/val2017/000000039769.jpg', 'http://images.cocodataset.org/val2017/000000039769.jpg'])
        self.assertEqual(nested_simplify(outputs, decimals=4), [[{'score': 0.9982, 'label': 'remote', 'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}}, {'score': 0.996, 'label': 'remote', 'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}}, {'score': 0.9955, 'label': 'couch', 'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}}, {'score': 0.9988, 'label': 'cat', 'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}}, {'score': 0.9987, 'label': 'cat', 'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}], [{'score': 0.9982, 'label': 'remote', 'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}}, {'score': 0.996, 'label': 'remote', 'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}}, {'score': 0.9955, 'label': 'couch', 'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}}, {'score': 0.9988, 'label': 'cat', 'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}}, {'score': 0.9987, 'label': 'cat', 'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}]])

    @require_torch
    @slow
    def test_integration_torch_object_detection(self):
        if False:
            print('Hello World!')
        model_id = 'facebook/detr-resnet-50'
        object_detector = pipeline('object-detection', model=model_id)
        outputs = object_detector('http://images.cocodataset.org/val2017/000000039769.jpg')
        self.assertEqual(nested_simplify(outputs, decimals=4), [{'score': 0.9982, 'label': 'remote', 'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}}, {'score': 0.996, 'label': 'remote', 'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}}, {'score': 0.9955, 'label': 'couch', 'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}}, {'score': 0.9988, 'label': 'cat', 'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}}, {'score': 0.9987, 'label': 'cat', 'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}])
        outputs = object_detector(['http://images.cocodataset.org/val2017/000000039769.jpg', 'http://images.cocodataset.org/val2017/000000039769.jpg'])
        self.assertEqual(nested_simplify(outputs, decimals=4), [[{'score': 0.9982, 'label': 'remote', 'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}}, {'score': 0.996, 'label': 'remote', 'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}}, {'score': 0.9955, 'label': 'couch', 'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}}, {'score': 0.9988, 'label': 'cat', 'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}}, {'score': 0.9987, 'label': 'cat', 'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}], [{'score': 0.9982, 'label': 'remote', 'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}}, {'score': 0.996, 'label': 'remote', 'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}}, {'score': 0.9955, 'label': 'couch', 'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}}, {'score': 0.9988, 'label': 'cat', 'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}}, {'score': 0.9987, 'label': 'cat', 'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}]])

    @require_torch
    @slow
    def test_threshold(self):
        if False:
            print('Hello World!')
        threshold = 0.9985
        model_id = 'facebook/detr-resnet-50'
        object_detector = pipeline('object-detection', model=model_id)
        outputs = object_detector('http://images.cocodataset.org/val2017/000000039769.jpg', threshold=threshold)
        self.assertEqual(nested_simplify(outputs, decimals=4), [{'score': 0.9988, 'label': 'cat', 'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}}, {'score': 0.9987, 'label': 'cat', 'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}])

    @require_torch
    @require_pytesseract
    @slow
    def test_layoutlm(self):
        if False:
            print('Hello World!')
        model_id = 'Narsil/layoutlmv3-finetuned-funsd'
        threshold = 0.9993
        object_detector = pipeline('object-detection', model=model_id, threshold=threshold)
        outputs = object_detector('https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png')
        self.assertEqual(nested_simplify(outputs, decimals=4), [{'score': 0.9993, 'label': 'I-ANSWER', 'box': {'xmin': 294, 'ymin': 254, 'xmax': 343, 'ymax': 264}}, {'score': 0.9993, 'label': 'I-ANSWER', 'box': {'xmin': 294, 'ymin': 254, 'xmax': 343, 'ymax': 264}}])