import unittest
from datasets import load_dataset
from transformers.pipelines import pipeline
from transformers.testing_utils import is_pipeline_test, nested_simplify, require_torch, slow

@is_pipeline_test
@require_torch
class ZeroShotAudioClassificationPipelineTests(unittest.TestCase):

    @require_torch
    def test_small_model_pt(self):
        if False:
            return 10
        audio_classifier = pipeline(task='zero-shot-audio-classification', model='hf-internal-testing/tiny-clap-htsat-unfused')
        dataset = load_dataset('ashraq/esc50')
        audio = dataset['train']['audio'][-1]['array']
        output = audio_classifier(audio, candidate_labels=['Sound of a dog', 'Sound of vaccum cleaner'])
        self.assertEqual(nested_simplify(output), [{'score': 0.501, 'label': 'Sound of a dog'}, {'score': 0.499, 'label': 'Sound of vaccum cleaner'}])

    @unittest.skip('No models are available in TF')
    def test_small_model_tf(self):
        if False:
            i = 10
            return i + 15
        pass

    @slow
    @require_torch
    def test_large_model_pt(self):
        if False:
            while True:
                i = 10
        audio_classifier = pipeline(task='zero-shot-audio-classification', model='laion/clap-htsat-unfused')
        dataset = load_dataset('ashraq/esc50')
        audio = dataset['train']['audio'][-1]['array']
        output = audio_classifier(audio, candidate_labels=['Sound of a dog', 'Sound of vaccum cleaner'])
        self.assertEqual(nested_simplify(output), [{'score': 0.999, 'label': 'Sound of a dog'}, {'score': 0.001, 'label': 'Sound of vaccum cleaner'}])
        output = audio_classifier([audio] * 5, candidate_labels=['Sound of a dog', 'Sound of vaccum cleaner'])
        self.assertEqual(nested_simplify(output), [[{'score': 0.999, 'label': 'Sound of a dog'}, {'score': 0.001, 'label': 'Sound of vaccum cleaner'}]] * 5)
        output = audio_classifier([audio] * 5, candidate_labels=['Sound of a dog', 'Sound of vaccum cleaner'], batch_size=5)
        self.assertEqual(nested_simplify(output), [[{'score': 0.999, 'label': 'Sound of a dog'}, {'score': 0.001, 'label': 'Sound of vaccum cleaner'}]] * 5)

    @unittest.skip('No models are available in TF')
    def test_large_model_tf(self):
        if False:
            while True:
                i = 10
        pass