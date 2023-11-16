import unittest
import numpy as np
from transformers import MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING, AutoProcessor, TextToAudioPipeline, pipeline
from transformers.testing_utils import is_pipeline_test, require_torch, require_torch_accelerator, require_torch_or_tf, slow, torch_device
from transformers.trainer_utils import set_seed
from .test_pipelines_common import ANY

@is_pipeline_test
@require_torch_or_tf
class TextToAudioPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING

    @slow
    @require_torch
    def test_small_musicgen_pt(self):
        if False:
            for i in range(10):
                print('nop')
        music_generator = pipeline(task='text-to-audio', model='facebook/musicgen-small', framework='pt')
        forward_params = {'do_sample': False, 'max_new_tokens': 250}
        outputs = music_generator('This is a test', forward_params=forward_params)
        self.assertEqual({'audio': ANY(np.ndarray), 'sampling_rate': 32000}, outputs)
        outputs = music_generator(['This is a test', 'This is a second test'], forward_params=forward_params)
        audio = [output['audio'] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)
        outputs = music_generator(['This is a test', 'This is a second test'], forward_params=forward_params, batch_size=2)
        audio = [output['audio'] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)

    @slow
    @require_torch
    def test_small_bark_pt(self):
        if False:
            for i in range(10):
                print('nop')
        speech_generator = pipeline(task='text-to-audio', model='suno/bark-small', framework='pt')
        forward_params = {'do_sample': False, 'semantic_max_new_tokens': 100}
        outputs = speech_generator('This is a test', forward_params=forward_params)
        self.assertEqual({'audio': ANY(np.ndarray), 'sampling_rate': 24000}, outputs)
        outputs = speech_generator(['This is a test', 'This is a second test'], forward_params=forward_params)
        audio = [output['audio'] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)
        forward_params = {'do_sample': True, 'semantic_max_new_tokens': 100, 'semantic_num_return_sequences': 2}
        outputs = speech_generator('This is a test', forward_params=forward_params)
        audio = outputs['audio']
        self.assertEqual(ANY(np.ndarray), audio)
        processor = AutoProcessor.from_pretrained('suno/bark-small')
        temp_inp = processor('hey, how are you?', voice_preset='v2/en_speaker_5')
        history_prompt = temp_inp['history_prompt']
        forward_params['history_prompt'] = history_prompt
        outputs = speech_generator(['This is a test', 'This is a second test'], forward_params=forward_params, batch_size=2)
        audio = [output['audio'] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)

    @slow
    @require_torch_accelerator
    def test_conversion_additional_tensor(self):
        if False:
            for i in range(10):
                print('nop')
        speech_generator = pipeline(task='text-to-audio', model='suno/bark-small', framework='pt', device=torch_device)
        processor = AutoProcessor.from_pretrained('suno/bark-small')
        forward_params = {'do_sample': True, 'semantic_max_new_tokens': 100}
        preprocess_params = {'max_length': 256, 'add_special_tokens': False, 'return_attention_mask': True, 'return_token_type_ids': False, 'padding': 'max_length'}
        outputs = speech_generator('This is a test', forward_params=forward_params, preprocess_params=preprocess_params)
        temp_inp = processor('hey, how are you?', voice_preset='v2/en_speaker_5')
        history_prompt = temp_inp['history_prompt']
        forward_params['history_prompt'] = history_prompt
        outputs = speech_generator('This is a test', forward_params=forward_params, preprocess_params=preprocess_params)
        self.assertEqual({'audio': ANY(np.ndarray), 'sampling_rate': 24000}, outputs)

    @slow
    @require_torch
    def test_vits_model_pt(self):
        if False:
            return 10
        speech_generator = pipeline(task='text-to-audio', model='facebook/mms-tts-eng', framework='pt')
        outputs = speech_generator('This is a test')
        self.assertEqual(outputs['sampling_rate'], 16000)
        audio = outputs['audio']
        self.assertEqual(ANY(np.ndarray), audio)
        outputs = speech_generator(['This is a test', 'This is a second test'])
        audio = [output['audio'] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)
        outputs = speech_generator(['This is a test', 'This is a second test'], batch_size=2)
        self.assertEqual(ANY(np.ndarray), outputs[0]['audio'])

    @slow
    @require_torch
    def test_forward_model_kwargs(self):
        if False:
            print('Hello World!')
        speech_generator = pipeline(task='text-to-audio', model='kakao-enterprise/vits-vctk', framework='pt')
        set_seed(555)
        outputs = speech_generator('This is a test', forward_params={'speaker_id': 5})
        audio = outputs['audio']
        with self.assertRaises(TypeError):
            outputs = speech_generator('This is a test', forward_params={'speaker_id': 5, 'do_sample': True})
        forward_params = {'speaker_id': 5}
        generate_kwargs = {'do_sample': True}
        with self.assertRaises(ValueError):
            outputs = speech_generator('This is a test', forward_params=forward_params, generate_kwargs=generate_kwargs)
        self.assertTrue(np.abs(outputs['audio'] - audio).max() < 1e-05)

    @slow
    @require_torch
    def test_generative_model_kwargs(self):
        if False:
            print('Hello World!')
        music_generator = pipeline(task='text-to-audio', model='facebook/musicgen-small', framework='pt')
        forward_params = {'do_sample': True, 'max_new_tokens': 250}
        set_seed(555)
        outputs = music_generator('This is a test', forward_params=forward_params)
        audio = outputs['audio']
        self.assertEqual(ANY(np.ndarray), audio)
        forward_params = {'do_sample': False, 'max_new_tokens': 250}
        generate_kwargs = {'do_sample': True}
        set_seed(555)
        outputs = music_generator('This is a test', forward_params=forward_params, generate_kwargs=generate_kwargs)
        self.assertListEqual(outputs['audio'].tolist(), audio.tolist())

    def get_test_pipeline(self, model, tokenizer, processor):
        if False:
            for i in range(10):
                print('nop')
        speech_generator = TextToAudioPipeline(model=model, tokenizer=tokenizer)
        return (speech_generator, ['This is a test', 'Another test'])

    def run_pipeline_test(self, speech_generator, _):
        if False:
            while True:
                i = 10
        outputs = speech_generator('This is a test')
        self.assertEqual(ANY(np.ndarray), outputs['audio'])
        forward_params = {'num_return_sequences': 2, 'do_sample': True} if speech_generator.model.can_generate() else {}
        outputs = speech_generator(['This is great !', 'Something else'], forward_params=forward_params)
        audio = [output['audio'] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)