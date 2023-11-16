import gc
import unittest
from transformers import MODEL_FOR_MASKED_LM_MAPPING, TF_MODEL_FOR_MASKED_LM_MAPPING, FillMaskPipeline, pipeline
from transformers.pipelines import PipelineException
from transformers.testing_utils import backend_empty_cache, is_pipeline_test, is_torch_available, nested_simplify, require_tf, require_torch, require_torch_accelerator, slow, torch_device
from .test_pipelines_common import ANY

@is_pipeline_test
class FillMaskPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_MASKED_LM_MAPPING
    tf_model_mapping = TF_MODEL_FOR_MASKED_LM_MAPPING

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        gc.collect()
        if is_torch_available():
            backend_empty_cache(torch_device)

    @require_tf
    def test_small_model_tf(self):
        if False:
            while True:
                i = 10
        unmasker = pipeline(task='fill-mask', model='sshleifer/tiny-distilroberta-base', top_k=2, framework='tf')
        outputs = unmasker('My name is <mask>')
        self.assertEqual(nested_simplify(outputs, decimals=6), [{'sequence': 'My name is grouped', 'score': 2.1e-05, 'token': 38015, 'token_str': ' grouped'}, {'sequence': 'My name is accuser', 'score': 2.1e-05, 'token': 25506, 'token_str': ' accuser'}])
        outputs = unmasker('The largest city in France is <mask>')
        self.assertEqual(nested_simplify(outputs, decimals=6), [{'sequence': 'The largest city in France is grouped', 'score': 2.1e-05, 'token': 38015, 'token_str': ' grouped'}, {'sequence': 'The largest city in France is accuser', 'score': 2.1e-05, 'token': 25506, 'token_str': ' accuser'}])
        outputs = unmasker('My name is <mask>', targets=[' Patrick', ' Clara', ' Teven'], top_k=3)
        self.assertEqual(nested_simplify(outputs, decimals=6), [{'sequence': 'My name is Clara', 'score': 2e-05, 'token': 13606, 'token_str': ' Clara'}, {'sequence': 'My name is Patrick', 'score': 2e-05, 'token': 3499, 'token_str': ' Patrick'}, {'sequence': 'My name is Te', 'score': 1.9e-05, 'token': 2941, 'token_str': ' Te'}])

    @require_torch
    def test_small_model_pt(self):
        if False:
            return 10
        unmasker = pipeline(task='fill-mask', model='sshleifer/tiny-distilroberta-base', top_k=2, framework='pt')
        outputs = unmasker('My name is <mask>')
        self.assertEqual(nested_simplify(outputs, decimals=6), [{'sequence': 'My name is Maul', 'score': 2.2e-05, 'token': 35676, 'token_str': ' Maul'}, {'sequence': 'My name isELS', 'score': 2.2e-05, 'token': 16416, 'token_str': 'ELS'}])
        outputs = unmasker('The largest city in France is <mask>')
        self.assertEqual(nested_simplify(outputs, decimals=6), [{'sequence': 'The largest city in France is Maul', 'score': 2.2e-05, 'token': 35676, 'token_str': ' Maul'}, {'sequence': 'The largest city in France isELS', 'score': 2.2e-05, 'token': 16416, 'token_str': 'ELS'}])
        outputs = unmasker('My name is <mask>', targets=[' Patrick', ' Clara', ' Teven'], top_k=3)
        self.assertEqual(nested_simplify(outputs, decimals=6), [{'sequence': 'My name is Patrick', 'score': 2.1e-05, 'token': 3499, 'token_str': ' Patrick'}, {'sequence': 'My name is Te', 'score': 2e-05, 'token': 2941, 'token_str': ' Te'}, {'sequence': 'My name is Clara', 'score': 2e-05, 'token': 13606, 'token_str': ' Clara'}])
        outputs = unmasker('My name is <mask> <mask>', top_k=2)
        self.assertEqual(nested_simplify(outputs, decimals=6), [[{'score': 2.2e-05, 'token': 35676, 'token_str': ' Maul', 'sequence': '<s>My name is Maul<mask></s>'}, {'score': 2.2e-05, 'token': 16416, 'token_str': 'ELS', 'sequence': '<s>My name isELS<mask></s>'}], [{'score': 2.2e-05, 'token': 35676, 'token_str': ' Maul', 'sequence': '<s>My name is<mask> Maul</s>'}, {'score': 2.2e-05, 'token': 16416, 'token_str': 'ELS', 'sequence': '<s>My name is<mask>ELS</s>'}]])

    @require_torch_accelerator
    def test_fp16_casting(self):
        if False:
            return 10
        pipe = pipeline('fill-mask', model='hf-internal-testing/tiny-random-distilbert', device=torch_device, framework='pt')
        pipe.model.half()
        response = pipe('Paris is the [MASK] of France.')
        self.assertIsInstance(response, list)

    @slow
    @require_torch
    def test_large_model_pt(self):
        if False:
            i = 10
            return i + 15
        unmasker = pipeline(task='fill-mask', model='distilroberta-base', top_k=2, framework='pt')
        self.run_large_test(unmasker)

    @slow
    @require_tf
    def test_large_model_tf(self):
        if False:
            while True:
                i = 10
        unmasker = pipeline(task='fill-mask', model='distilroberta-base', top_k=2, framework='tf')
        self.run_large_test(unmasker)

    def run_large_test(self, unmasker):
        if False:
            return 10
        outputs = unmasker('My name is <mask>')
        self.assertEqual(nested_simplify(outputs), [{'sequence': 'My name is John', 'score': 0.008, 'token': 610, 'token_str': ' John'}, {'sequence': 'My name is Chris', 'score': 0.007, 'token': 1573, 'token_str': ' Chris'}])
        outputs = unmasker('The largest city in France is <mask>')
        self.assertEqual(nested_simplify(outputs), [{'sequence': 'The largest city in France is Paris', 'score': 0.251, 'token': 2201, 'token_str': ' Paris'}, {'sequence': 'The largest city in France is Lyon', 'score': 0.214, 'token': 12790, 'token_str': ' Lyon'}])
        outputs = unmasker('My name is <mask>', targets=[' Patrick', ' Clara', ' Teven'], top_k=3)
        self.assertEqual(nested_simplify(outputs), [{'sequence': 'My name is Patrick', 'score': 0.005, 'token': 3499, 'token_str': ' Patrick'}, {'sequence': 'My name is Clara', 'score': 0.0, 'token': 13606, 'token_str': ' Clara'}, {'sequence': 'My name is Te', 'score': 0.0, 'token': 2941, 'token_str': ' Te'}])
        outputs = unmasker('My name is <mask>' + 'Lorem ipsum dolor sit amet, consectetur adipiscing elit,' * 100, tokenizer_kwargs={'truncation': True})
        self.assertEqual(nested_simplify(outputs, decimals=6), [{'sequence': 'My name is grouped', 'score': 2.2e-05, 'token': 38015, 'token_str': ' grouped'}, {'sequence': 'My name is accuser', 'score': 2.1e-05, 'token': 25506, 'token_str': ' accuser'}])

    @require_torch
    def test_model_no_pad_pt(self):
        if False:
            for i in range(10):
                print('nop')
        unmasker = pipeline(task='fill-mask', model='sshleifer/tiny-distilroberta-base', framework='pt')
        unmasker.tokenizer.pad_token_id = None
        unmasker.tokenizer.pad_token = None
        self.run_pipeline_test(unmasker, [])

    @require_tf
    def test_model_no_pad_tf(self):
        if False:
            return 10
        unmasker = pipeline(task='fill-mask', model='sshleifer/tiny-distilroberta-base', framework='tf')
        unmasker.tokenizer.pad_token_id = None
        unmasker.tokenizer.pad_token = None
        self.run_pipeline_test(unmasker, [])

    def get_test_pipeline(self, model, tokenizer, processor):
        if False:
            print('Hello World!')
        if tokenizer is None or tokenizer.mask_token_id is None:
            self.skipTest('The provided tokenizer has no mask token, (probably reformer or wav2vec2)')
        fill_masker = FillMaskPipeline(model=model, tokenizer=tokenizer)
        examples = [f'This is another {tokenizer.mask_token} test']
        return (fill_masker, examples)

    def run_pipeline_test(self, fill_masker, examples):
        if False:
            return 10
        tokenizer = fill_masker.tokenizer
        model = fill_masker.model
        outputs = fill_masker(f'This is a {tokenizer.mask_token}')
        self.assertEqual(outputs, [{'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}])
        outputs = fill_masker([f'This is a {tokenizer.mask_token}'])
        self.assertEqual(outputs, [{'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}])
        outputs = fill_masker([f'This is a {tokenizer.mask_token}', f'Another {tokenizer.mask_token} great test.'])
        self.assertEqual(outputs, [[{'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}], [{'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}]])
        with self.assertRaises(ValueError):
            fill_masker([None])
        with self.assertRaises(PipelineException):
            fill_masker('This is')
        self.run_test_top_k(model, tokenizer)
        self.run_test_targets(model, tokenizer)
        self.run_test_top_k_targets(model, tokenizer)
        self.fill_mask_with_duplicate_targets_and_top_k(model, tokenizer)
        self.fill_mask_with_multiple_masks(model, tokenizer)

    def run_test_targets(self, model, tokenizer):
        if False:
            print('Hello World!')
        vocab = tokenizer.get_vocab()
        targets = sorted(vocab.keys())[:2]
        fill_masker = FillMaskPipeline(model=model, tokenizer=tokenizer, targets=targets)
        outputs = fill_masker(f'This is a {tokenizer.mask_token}')
        self.assertEqual(outputs, [{'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}])
        target_ids = {vocab[el] for el in targets}
        self.assertEqual({el['token'] for el in outputs}, target_ids)
        processed_targets = [tokenizer.decode([x]) for x in target_ids]
        self.assertEqual({el['token_str'] for el in outputs}, set(processed_targets))
        fill_masker = FillMaskPipeline(model=model, tokenizer=tokenizer)
        outputs = fill_masker(f'This is a {tokenizer.mask_token}', targets=targets)
        self.assertEqual(outputs, [{'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}])
        target_ids = {vocab[el] for el in targets}
        self.assertEqual({el['token'] for el in outputs}, target_ids)
        processed_targets = [tokenizer.decode([x]) for x in target_ids]
        self.assertEqual({el['token_str'] for el in outputs}, set(processed_targets))
        outputs = fill_masker(f'This is a {tokenizer.mask_token}', targets=targets)
        tokens = [top_mask['token_str'] for top_mask in outputs]
        scores = [top_mask['score'] for top_mask in outputs]
        if set(tokens) == set(targets):
            unmasked_targets = fill_masker(f'This is a {tokenizer.mask_token}', targets=tokens)
            target_scores = [top_mask['score'] for top_mask in unmasked_targets]
            self.assertEqual(nested_simplify(scores), nested_simplify(target_scores))
        with self.assertRaises(ValueError):
            outputs = fill_masker(f'This is a {tokenizer.mask_token}', targets=[])
        if '' not in tokenizer.get_vocab():
            with self.assertRaises(ValueError):
                outputs = fill_masker(f'This is a {tokenizer.mask_token}', targets=[''])
            with self.assertRaises(ValueError):
                outputs = fill_masker(f'This is a {tokenizer.mask_token}', targets='')

    def run_test_top_k(self, model, tokenizer):
        if False:
            for i in range(10):
                print('nop')
        fill_masker = FillMaskPipeline(model=model, tokenizer=tokenizer, top_k=2)
        outputs = fill_masker(f'This is a {tokenizer.mask_token}')
        self.assertEqual(outputs, [{'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}])
        fill_masker = FillMaskPipeline(model=model, tokenizer=tokenizer)
        outputs2 = fill_masker(f'This is a {tokenizer.mask_token}', top_k=2)
        self.assertEqual(outputs2, [{'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}])
        self.assertEqual(nested_simplify(outputs), nested_simplify(outputs2))

    def run_test_top_k_targets(self, model, tokenizer):
        if False:
            i = 10
            return i + 15
        vocab = tokenizer.get_vocab()
        fill_masker = FillMaskPipeline(model=model, tokenizer=tokenizer)
        targets = sorted(vocab.keys())[:3]
        outputs = fill_masker(f'This is a {tokenizer.mask_token}', top_k=2, targets=targets)
        targets2 = [el['token_str'] for el in sorted(outputs, key=lambda x: x['score'], reverse=True)]
        if set(targets2).issubset(targets):
            outputs2 = fill_masker(f'This is a {tokenizer.mask_token}', top_k=3, targets=targets2)
            self.assertEqual(nested_simplify(outputs), nested_simplify(outputs2))

    def fill_mask_with_duplicate_targets_and_top_k(self, model, tokenizer):
        if False:
            while True:
                i = 10
        fill_masker = FillMaskPipeline(model=model, tokenizer=tokenizer)
        vocab = tokenizer.get_vocab()
        targets = sorted(vocab.keys())[:3]
        targets = [targets[0], targets[1], targets[0], targets[2], targets[1]]
        outputs = fill_masker(f'My name is {tokenizer.mask_token}', targets=targets, top_k=10)
        self.assertEqual(len(outputs), 3)

    def fill_mask_with_multiple_masks(self, model, tokenizer):
        if False:
            for i in range(10):
                print('nop')
        fill_masker = FillMaskPipeline(model=model, tokenizer=tokenizer)
        outputs = fill_masker(f'This is a {tokenizer.mask_token} {tokenizer.mask_token} {tokenizer.mask_token}', top_k=2)
        self.assertEqual(outputs, [[{'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}], [{'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}], [{'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}, {'sequence': ANY(str), 'score': ANY(float), 'token': ANY(int), 'token_str': ANY(str)}]])