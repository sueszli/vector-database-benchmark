from __future__ import annotations
import os
import tempfile
import unittest
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import is_tensorflow_text_available, is_tf_available
from transformers.testing_utils import require_tensorflow_text, require_tf, slow
from ..test_modeling_tf_common import floats_tensor
from .test_framework_agnostic import GenerationIntegrationTestsMixin
if is_tf_available():
    import tensorflow as tf
    from transformers import AutoTokenizer, TFAutoModelForCausalLM, TFAutoModelForSeq2SeqLM, TFAutoModelForSpeechSeq2Seq, TFAutoModelForVision2Seq, TFBartForConditionalGeneration, TFLogitsProcessorList, TFMinLengthLogitsProcessor, tf_top_k_top_p_filtering
if is_tensorflow_text_available():
    import tensorflow_text as text

@require_tf
class UtilsFunctionsTest(unittest.TestCase):

    def test_top_k_top_p_filtering(self):
        if False:
            while True:
                i = 10
        logits = tf.convert_to_tensor([[8.2220991, -0.5620044, 5.23229752, 4.0386393, -6.8798378, -0.54785802, -3.2012153, 2.92777176, 1.88171953, 7.35341276, 8.43207833, -9.85711836, -5.96209236, -1.13039161, -7.1115294, -0.8369633, -5.3186408, 7.06427407, 0.81369344, -0.82023817, -5.9179796, 0.58813443, -6.99778438, 4.71551189, -0.18771637, 7.44020759, 9.38450987, 2.12662941, -9.32562038, 2.35652522], [0.58425518, 4.53139238, -5.57510464, -6.28030699, -7.19529503, -4.02122551, 1.39337037, -6.06707057, 1.59480517, -9.643119, 0.03907799, 0.67231762, -8.88206726, 6.27115922, 2.28520723, 4.82767506, 4.30421368, 8.8275313, 5.44029958, -4.4735794, 7.38579536, -2.91051663, 2.61946077, -2.5674762, -9.48959302, -4.02922645, -1.35416918, 9.67702323, -5.89478553, 1.85370467]], dtype=tf.float32)
        non_inf_expected_idx = tf.convert_to_tensor([[0, 0], [0, 9], [0, 10], [0, 25], [0, 26], [1, 13], [1, 17], [1, 18], [1, 20], [1, 27]], dtype=tf.int32)
        non_inf_expected_output = tf.convert_to_tensor([8.222099, 7.3534126, 8.432078, 7.4402075, 9.38451, 6.271159, 8.827531, 5.4402995, 7.3857956, 9.677023], dtype=tf.float32)
        output = tf_top_k_top_p_filtering(logits, top_k=10, top_p=0.6, min_tokens_to_keep=4)
        non_inf_output = output[output != -float('inf')]
        non_inf_idx = tf.cast(tf.where(tf.not_equal(output, tf.constant(-float('inf'), dtype=tf.float32))), dtype=tf.int32)
        tf.debugging.assert_near(non_inf_output, non_inf_expected_output, rtol=1e-12)
        tf.debugging.assert_equal(non_inf_idx, non_inf_expected_idx)

@require_tf
class TFGenerationIntegrationTests(unittest.TestCase, GenerationIntegrationTestsMixin):
    if is_tf_available():
        framework_dependent_parameters = {'AutoModelForCausalLM': TFAutoModelForCausalLM, 'AutoModelForSpeechSeq2Seq': TFAutoModelForSpeechSeq2Seq, 'AutoModelForSeq2SeqLM': TFAutoModelForSeq2SeqLM, 'AutoModelForVision2Seq': TFAutoModelForVision2Seq, 'LogitsProcessorList': TFLogitsProcessorList, 'MinLengthLogitsProcessor': TFMinLengthLogitsProcessor, 'create_tensor_fn': tf.convert_to_tensor, 'floats_tensor': floats_tensor, 'return_tensors': 'tf'}

    @slow
    def test_generate_tf_function_export_fixed_input_length(self):
        if False:
            while True:
                i = 10
        test_model = TFAutoModelForCausalLM.from_pretrained('hf-internal-testing/tiny-random-gpt2')
        input_length = 2
        max_new_tokens = 2

        class DummyModel(tf.Module):

            def __init__(self, model):
                if False:
                    print('Hello World!')
                super(DummyModel, self).__init__()
                self.model = model

            @tf.function(input_signature=(tf.TensorSpec((None, input_length), tf.int32, name='input_ids'), tf.TensorSpec((None, input_length), tf.int32, name='attention_mask')), jit_compile=True)
            def serving(self, input_ids, attention_mask):
                if False:
                    i = 10
                    return i + 15
                outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, return_dict_in_generate=True)
                return {'sequences': outputs['sequences']}
        dummy_input_ids = [[2, 0], [102, 103]]
        dummy_attention_masks = [[1, 0], [1, 1]]
        dummy_model = DummyModel(model=test_model)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tf.saved_model.save(dummy_model, tmp_dir, signatures={'serving_default': dummy_model.serving})
            serving_func = tf.saved_model.load(tmp_dir).signatures['serving_default']
            for batch_size in range(1, len(dummy_input_ids) + 1):
                inputs = {'input_ids': tf.constant(dummy_input_ids[:batch_size]), 'attention_mask': tf.constant(dummy_attention_masks[:batch_size])}
                tf_func_outputs = serving_func(**inputs)['sequences']
                tf_model_outputs = test_model.generate(**inputs, max_new_tokens=max_new_tokens)
                tf.debugging.assert_equal(tf_func_outputs, tf_model_outputs)

    @slow
    def test_generate_tf_function_export_fixed_batch_size(self):
        if False:
            i = 10
            return i + 15
        test_model = TFAutoModelForCausalLM.from_pretrained('hf-internal-testing/tiny-random-gpt2')
        batch_size = 1
        max_new_tokens = 2

        class DummyModel(tf.Module):

            def __init__(self, model):
                if False:
                    for i in range(10):
                        print('nop')
                super(DummyModel, self).__init__()
                self.model = model

            @tf.function(input_signature=(tf.TensorSpec((batch_size, None), tf.int32, name='input_ids'), tf.TensorSpec((batch_size, None), tf.int32, name='attention_mask')), jit_compile=True)
            def serving(self, input_ids, attention_mask):
                if False:
                    i = 10
                    return i + 15
                outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, return_dict_in_generate=True)
                return {'sequences': outputs['sequences']}
        dummy_input_ids = [[2], [102, 103]]
        dummy_attention_masks = [[1], [1, 1]]
        dummy_model = DummyModel(model=test_model)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tf.saved_model.save(dummy_model, tmp_dir, signatures={'serving_default': dummy_model.serving})
            serving_func = tf.saved_model.load(tmp_dir).signatures['serving_default']
            for input_row in range(len(dummy_input_ids)):
                inputs = {'input_ids': tf.constant([dummy_input_ids[input_row]]), 'attention_mask': tf.constant([dummy_attention_masks[input_row]])}
                tf_func_outputs = serving_func(**inputs)['sequences']
                tf_model_outputs = test_model.generate(**inputs, max_new_tokens=max_new_tokens)
                tf.debugging.assert_equal(tf_func_outputs, tf_model_outputs)

    @slow
    @require_tensorflow_text
    def test_generate_tf_function_export_with_tf_tokenizer(self):
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as tmp_dir:
            hf_hub_download(repo_id='google/flan-t5-small', filename='spiece.model', local_dir=tmp_dir)

            class CompleteSentenceTransformer(tf.keras.layers.Layer):

                def __init__(self):
                    if False:
                        return 10
                    super().__init__()
                    self.tokenizer = text.SentencepieceTokenizer(model=tf.io.gfile.GFile(os.path.join(tmp_dir, 'spiece.model'), 'rb').read())
                    self.model = TFAutoModelForSeq2SeqLM.from_pretrained('hf-internal-testing/tiny-random-t5')

                def call(self, inputs, *args, **kwargs):
                    if False:
                        for i in range(10):
                            print('nop')
                    tokens = self.tokenizer.tokenize(inputs)
                    (input_ids, attention_mask) = text.pad_model_inputs(tokens, max_seq_length=64, pad_value=self.model.config.pad_token_id)
                    outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
                    return self.tokenizer.detokenize(outputs)
            complete_model = CompleteSentenceTransformer()
            inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='inputs')
            outputs = complete_model(inputs)
            keras_model = tf.keras.Model(inputs, outputs)
            keras_model.save(tmp_dir)

    def test_eos_token_id_int_and_list_top_k_top_sampling(self):
        if False:
            print('Hello World!')
        generation_kwargs = {'do_sample': True, 'num_beams': 1, 'top_p': 0.7, 'top_k': 10, 'temperature': 0.7}
        expectation = 14
        tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/tiny-random-gpt2')
        text = 'Hello, my dog is cute and'
        tokens = tokenizer(text, return_tensors='tf')
        model = TFAutoModelForCausalLM.from_pretrained('hf-internal-testing/tiny-random-gpt2')
        eos_token_id = 638
        with tf.device(':/CPU:0'):
            tf.random.set_seed(0)
            generated_tokens = model.generate(**tokens, eos_token_id=eos_token_id, **generation_kwargs)
        self.assertTrue(expectation == len(generated_tokens[0]))
        eos_token_id = [638, 198]
        with tf.device(':/CPU:0'):
            tf.random.set_seed(0)
            generated_tokens = model.generate(**tokens, eos_token_id=eos_token_id, **generation_kwargs)
        self.assertTrue(expectation == len(generated_tokens[0]))

    def test_model_kwarg_encoder_signature_filtering(self):
        if False:
            for i in range(10):
                print('nop')
        bart_tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/tiny-random-bart')
        article = 'Hugging Face is a technology company based in New York and Paris.'
        input_ids = bart_tokenizer(article, return_tensors='tf').input_ids
        bart_model = TFBartForConditionalGeneration.from_pretrained('hf-internal-testing/tiny-random-bart')
        output = bart_model.generate(input_ids).numpy()

        class FakeBart(TFBartForConditionalGeneration):

            def call(self, input_ids, foo=None, **kwargs):
                if False:
                    return 10
                return super().call(input_ids, **kwargs)
        bart_model = FakeBart.from_pretrained('hf-internal-testing/tiny-random-bart')
        fake_output = bart_model.generate(input_ids, foo='bar').numpy()
        self.assertTrue(np.array_equal(output, fake_output))

        class FakeEncoder(bart_model.model.encoder.__class__):

            def call(self, input_ids, **kwargs):
                if False:
                    i = 10
                    return i + 15
                return super().call(input_ids, **kwargs)
        fake_encoder = FakeEncoder(bart_model.config, bart_model.model.shared)
        bart_model.model.encoder = fake_encoder
        fake_output = bart_model.generate(input_ids).numpy()
        with self.assertRaises(ValueError):
            bart_model.generate(input_ids, foo='bar')