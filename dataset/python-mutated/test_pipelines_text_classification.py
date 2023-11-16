import unittest
from transformers import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, TextClassificationPipeline, pipeline
from transformers.testing_utils import is_pipeline_test, nested_simplify, require_tf, require_torch, slow, torch_device
from .test_pipelines_common import ANY
_TO_SKIP = {'LayoutLMv2Config', 'LayoutLMv3Config'}

@is_pipeline_test
class TextClassificationPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
    tf_model_mapping = TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
    if model_mapping is not None:
        model_mapping = {config: model for (config, model) in model_mapping.items() if config.__name__ not in _TO_SKIP}
    if tf_model_mapping is not None:
        tf_model_mapping = {config: model for (config, model) in tf_model_mapping.items() if config.__name__ not in _TO_SKIP}

    @require_torch
    def test_small_model_pt(self):
        if False:
            i = 10
            return i + 15
        text_classifier = pipeline(task='text-classification', model='hf-internal-testing/tiny-random-distilbert', framework='pt')
        outputs = text_classifier('This is great !')
        self.assertEqual(nested_simplify(outputs), [{'label': 'LABEL_0', 'score': 0.504}])
        outputs = text_classifier('This is great !', top_k=2)
        self.assertEqual(nested_simplify(outputs), [{'label': 'LABEL_0', 'score': 0.504}, {'label': 'LABEL_1', 'score': 0.496}])
        outputs = text_classifier(['This is great !', 'This is bad'], top_k=2)
        self.assertEqual(nested_simplify(outputs), [[{'label': 'LABEL_0', 'score': 0.504}, {'label': 'LABEL_1', 'score': 0.496}], [{'label': 'LABEL_0', 'score': 0.504}, {'label': 'LABEL_1', 'score': 0.496}]])
        outputs = text_classifier('This is great !', top_k=1)
        self.assertEqual(nested_simplify(outputs), [{'label': 'LABEL_0', 'score': 0.504}])
        outputs = text_classifier('This is great !', return_all_scores=False)
        self.assertEqual(nested_simplify(outputs), [{'label': 'LABEL_0', 'score': 0.504}])
        outputs = text_classifier('This is great !', return_all_scores=True)
        self.assertEqual(nested_simplify(outputs), [[{'label': 'LABEL_0', 'score': 0.504}, {'label': 'LABEL_1', 'score': 0.496}]])
        outputs = text_classifier(['This is great !', 'Something else'], return_all_scores=True)
        self.assertEqual(nested_simplify(outputs), [[{'label': 'LABEL_0', 'score': 0.504}, {'label': 'LABEL_1', 'score': 0.496}], [{'label': 'LABEL_0', 'score': 0.504}, {'label': 'LABEL_1', 'score': 0.496}]])
        outputs = text_classifier(['This is great !', 'Something else'], return_all_scores=False)
        self.assertEqual(nested_simplify(outputs), [{'label': 'LABEL_0', 'score': 0.504}, {'label': 'LABEL_0', 'score': 0.504}])

    @require_torch
    def test_accepts_torch_device(self):
        if False:
            print('Hello World!')
        text_classifier = pipeline(task='text-classification', model='hf-internal-testing/tiny-random-distilbert', framework='pt', device=torch_device)
        outputs = text_classifier('This is great !')
        self.assertEqual(nested_simplify(outputs), [{'label': 'LABEL_0', 'score': 0.504}])

    @require_tf
    def test_small_model_tf(self):
        if False:
            return 10
        text_classifier = pipeline(task='text-classification', model='hf-internal-testing/tiny-random-distilbert', framework='tf')
        outputs = text_classifier('This is great !')
        self.assertEqual(nested_simplify(outputs), [{'label': 'LABEL_0', 'score': 0.504}])

    @slow
    @require_torch
    def test_pt_bert(self):
        if False:
            i = 10
            return i + 15
        text_classifier = pipeline('text-classification')
        outputs = text_classifier('This is great !')
        self.assertEqual(nested_simplify(outputs), [{'label': 'POSITIVE', 'score': 1.0}])
        outputs = text_classifier('This is bad !')
        self.assertEqual(nested_simplify(outputs), [{'label': 'NEGATIVE', 'score': 1.0}])
        outputs = text_classifier('Birds are a type of animal')
        self.assertEqual(nested_simplify(outputs), [{'label': 'POSITIVE', 'score': 0.988}])

    @slow
    @require_tf
    def test_tf_bert(self):
        if False:
            i = 10
            return i + 15
        text_classifier = pipeline('text-classification', framework='tf')
        outputs = text_classifier('This is great !')
        self.assertEqual(nested_simplify(outputs), [{'label': 'POSITIVE', 'score': 1.0}])
        outputs = text_classifier('This is bad !')
        self.assertEqual(nested_simplify(outputs), [{'label': 'NEGATIVE', 'score': 1.0}])
        outputs = text_classifier('Birds are a type of animal')
        self.assertEqual(nested_simplify(outputs), [{'label': 'POSITIVE', 'score': 0.988}])

    def get_test_pipeline(self, model, tokenizer, processor):
        if False:
            print('Hello World!')
        text_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)
        return (text_classifier, ['HuggingFace is in', 'This is another test'])

    def run_pipeline_test(self, text_classifier, _):
        if False:
            while True:
                i = 10
        model = text_classifier.model
        valid_inputs = 'HuggingFace is in'
        outputs = text_classifier(valid_inputs)
        self.assertEqual(nested_simplify(outputs), [{'label': ANY(str), 'score': ANY(float)}])
        self.assertTrue(outputs[0]['label'] in model.config.id2label.values())
        valid_inputs = ['HuggingFace is in ', 'Paris is in France']
        outputs = text_classifier(valid_inputs)
        self.assertEqual(nested_simplify(outputs), [{'label': ANY(str), 'score': ANY(float)}, {'label': ANY(str), 'score': ANY(float)}])
        self.assertTrue(outputs[0]['label'] in model.config.id2label.values())
        self.assertTrue(outputs[1]['label'] in model.config.id2label.values())
        outputs = text_classifier(valid_inputs, top_k=None)
        N = len(model.config.id2label.values())
        self.assertEqual(nested_simplify(outputs), [[{'label': ANY(str), 'score': ANY(float)}] * N, [{'label': ANY(str), 'score': ANY(float)}] * N])
        valid_inputs = {'text': 'HuggingFace is in ', 'text_pair': 'Paris is in France'}
        outputs = text_classifier(valid_inputs)
        self.assertEqual(nested_simplify(outputs), {'label': ANY(str), 'score': ANY(float)})
        self.assertTrue(outputs['label'] in model.config.id2label.values())
        invalid_input = [['HuggingFace is in ', 'Paris is in France']]
        with self.assertRaises(ValueError):
            text_classifier(invalid_input)
        outputs = text_classifier([[['HuggingFace is in ', 'Paris is in France']]])
        self.assertEqual(nested_simplify(outputs), [{'label': ANY(str), 'score': ANY(float)}])
        self.assertTrue(outputs[0]['label'] in model.config.id2label.values())