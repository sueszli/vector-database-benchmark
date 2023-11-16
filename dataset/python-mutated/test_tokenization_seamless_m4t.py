import tempfile
import unittest
from transformers import SPIECE_UNDERLINE, AddedToken, BatchEncoding, PreTrainedTokenizerFast, SeamlessM4TTokenizer, SeamlessM4TTokenizerFast, is_torch_available
from transformers.testing_utils import get_tests_dir, nested_simplify, require_sentencepiece, require_tokenizers, require_torch
from ...test_tokenization_common import TokenizerTesterMixin
SAMPLE_VOCAB = get_tests_dir('fixtures/test_sentencepiece.model')
if is_torch_available():
    from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right
EN_CODE = 256047
RO_CODE = 256145
SMALL_TRAINING_CORPUS = [['This is the first sentence.', 'This is the second one.'], ['This sentence (contains #) over symbols and numbers 12 3.', 'But not this one.']]

@require_sentencepiece
@require_tokenizers
class SeamlessM4TTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = SeamlessM4TTokenizer
    rust_tokenizer_class = SeamlessM4TTokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = True
    from_pretrained_kwargs = {}

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        tokenizer = SeamlessM4TTokenizer(SAMPLE_VOCAB, keep_accents=True)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_full_tokenizer(self):
        if False:
            print('Hello World!')
        tokenizer = SeamlessM4TTokenizer(SAMPLE_VOCAB, keep_accents=True)
        tokens = tokenizer.tokenize('This is a test')
        self.assertListEqual(tokens, ['▁This', '▁is', '▁a', '▁t', 'est'])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [value + tokenizer.fairseq_offset for value in [285, 46, 10, 170, 382]])
        tokens = tokenizer.tokenize('I was born in 92000, and this is falsé.')
        self.assertListEqual(tokens, [SPIECE_UNDERLINE + 'I', SPIECE_UNDERLINE + 'was', SPIECE_UNDERLINE + 'b', 'or', 'n', SPIECE_UNDERLINE + 'in', SPIECE_UNDERLINE + '', '9', '2', '0', '0', '0', ',', SPIECE_UNDERLINE + 'and', SPIECE_UNDERLINE + 'this', SPIECE_UNDERLINE + 'is', SPIECE_UNDERLINE + 'f', 'al', 's', 'é', '.'])
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [value + tokenizer.fairseq_offset for value in [8, 21, 84, 55, 24, 19, 7, 0, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 0, 4]])
        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(back_tokens, [SPIECE_UNDERLINE + 'I', SPIECE_UNDERLINE + 'was', SPIECE_UNDERLINE + 'b', 'or', 'n', SPIECE_UNDERLINE + 'in', SPIECE_UNDERLINE + '', '<unk>', '2', '0', '0', '0', ',', SPIECE_UNDERLINE + 'and', SPIECE_UNDERLINE + 'this', SPIECE_UNDERLINE + 'is', SPIECE_UNDERLINE + 'f', 'al', 's', '<unk>', '.'])

    def test_maximum_encoding_length_single_input(self):
        if False:
            for i in range(10):
                print('nop')
        tokenizers = self.get_tokenizers(do_lower_case=False, model_max_length=100)
        for tokenizer in tokenizers:
            with self.subTest(f'{tokenizer.__class__.__name__}'):
                (seq_0, ids) = self.get_clean_sequence(tokenizer, max_length=20)
                sequence = tokenizer.encode(seq_0, add_special_tokens=False)
                total_length = len(sequence)
                self.assertGreater(total_length, 4, "Issue with the testing sequence, please update it, it's too short")
                model_max_length = tokenizer.model_max_length
                self.assertEqual(model_max_length, 100)
                seq_1 = seq_0 * model_max_length
                sequence1 = tokenizer(seq_1, add_special_tokens=False)
                total_length1 = len(sequence1['input_ids'])
                self.assertGreater(total_length1, model_max_length, "Issue with the testing sequence, please update it, it's too short")
                padding_strategies = [False, True, 'longest'] if tokenizer.pad_token and tokenizer.pad_token_id >= 0 else [False]
                for padding_state in padding_strategies:
                    with self.subTest(f'Padding: {padding_state}'):
                        for truncation_state in [True, 'longest_first', 'only_first']:
                            with self.subTest(f'Truncation: {truncation_state}'):
                                output = tokenizer(seq_1, padding=padding_state, truncation=truncation_state)
                                self.assertEqual(len(output['input_ids']), model_max_length)
                                output = tokenizer([seq_1], padding=padding_state, truncation=truncation_state)
                                self.assertEqual(len(output['input_ids'][0]), model_max_length)
                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs('transformers', level='WARNING') as cm:
                            output = tokenizer(seq_1, padding=padding_state, truncation=False)
                            self.assertNotEqual(len(output['input_ids']), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(cm.records[0].message.startswith('Token indices sequence length is longer than the specified maximum sequence length for this model'))
                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs('transformers', level='WARNING') as cm:
                            output = tokenizer([seq_1], padding=padding_state, truncation=False)
                            self.assertNotEqual(len(output['input_ids'][0]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(cm.records[0].message.startswith('Token indices sequence length is longer than the specified maximum sequence length for this model'))
                stride = 2
                information = tokenizer(seq_0, max_length=total_length - 2, add_special_tokens=False, stride=stride, truncation='longest_first', return_overflowing_tokens=True, padding=False)
                if isinstance(tokenizer, PreTrainedTokenizerFast):
                    truncated_sequence = information['input_ids'][0]
                    overflowing_tokens = information['input_ids'][1]
                    self.assertEqual(len(information['input_ids']), 2)
                    self.assertEqual(len(truncated_sequence), total_length - 2)
                    self.assertEqual(truncated_sequence, sequence[:-2])
                    self.assertEqual(len(overflowing_tokens), 2 + stride)
                    self.assertEqual(overflowing_tokens, sequence[-(2 + stride):])
                else:
                    truncated_sequence = information['input_ids']
                    overflowing_tokens = information['overflowing_tokens']
                    self.assertEqual(len(truncated_sequence), total_length - 2)
                    self.assertEqual(truncated_sequence, sequence[:-2])
                    self.assertEqual(len(overflowing_tokens), 2 + stride)
                    self.assertEqual(overflowing_tokens, sequence[-(2 + stride):])

    @unittest.skip('By defaults, uses pad_to_multiple_of which breaks the test')
    def test_maximum_encoding_length_pair_input(self):
        if False:
            while True:
                i = 10
        pass

    def test_padding_to_multiple_of(self):
        if False:
            i = 10
            return i + 15
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f'{tokenizer.__class__.__name__}'):
                if tokenizer.pad_token is None:
                    self.skipTest('No padding token.')
                else:
                    empty_tokens = tokenizer('', padding=True, pad_to_multiple_of=8)
                    normal_tokens = tokenizer('This is a sample input', padding=True, pad_to_multiple_of=8)
                    for (key, value) in empty_tokens.items():
                        self.assertEqual(len(value) % 8, 0, f'BatchEncoding.{key} is not multiple of 8')
                    for (key, value) in normal_tokens.items():
                        self.assertEqual(len(value) % 8, 0, f'BatchEncoding.{key} is not multiple of 8')
                    normal_tokens = tokenizer('This', pad_to_multiple_of=8, padding=False)
                    for (key, value) in normal_tokens.items():
                        self.assertNotEqual(len(value) % 8, 0, f'BatchEncoding.{key} is not multiple of 8')
                    normal_tokens = tokenizer('This', padding=True, truncation=True, pad_to_multiple_of=8)
                    for (key, value) in normal_tokens.items():
                        self.assertEqual(len(value) % 8, 0, f'BatchEncoding.{key} is not multiple of 8')
                    self.assertRaises(ValueError, tokenizer.__call__, 'This', padding=True, truncation=True, max_length=12, pad_to_multiple_of=8)

    @require_torch
    def test_prepare_seq2seq_batch(self):
        if False:
            i = 10
            return i + 15
        if not self.test_seq2seq:
            return
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f'{tokenizer.__class__.__name__}'):
                src_text = [' UN Chief Says There Is No Military Solution in Syria', " Secretary-General Ban Ki-moon says his response to Russia's stepped up military support for Syria is that 'there is no military solution' to the nearly five-year conflict and more weapons will only worsen the violence and misery for millions of people."]
                tgt_text = ['Şeful ONU declară că nu există o soluţie militară în Siria', 'Secretarul General Ban Ki-moon declară că răspunsul său la intensificarea sprijinului militar al Rusiei pentru Siria este că "nu există o soluţie militară" la conflictul de aproape cinci ani şi că noi arme nu vor face decât să înrăutăţească violenţele şi mizeria pentru milioane de oameni.']
                try:
                    batch = tokenizer.prepare_seq2seq_batch(src_texts=src_text, tgt_texts=tgt_text, max_length=3, max_target_length=10, return_tensors='pt', src_lang='eng', tgt_lang='ron', pad_to_multiple_of=None)
                except NotImplementedError:
                    return
                self.assertEqual(batch.input_ids.shape[1], 3)
                self.assertEqual(batch.labels.shape[1], 10)
                batch = tokenizer.prepare_seq2seq_batch(src_texts=src_text, tgt_texts=tgt_text, max_length=4, return_tensors='pt', pad_to_multiple_of=None)
                self.assertEqual(batch.input_ids.shape[1], 4)
                self.assertEqual(batch.labels.shape[1], 4)
                batch_encoder_only = tokenizer.prepare_seq2seq_batch(src_texts=src_text, max_length=4, max_target_length=10, return_tensors='pt', pad_to_multiple_of=None)
                self.assertEqual(batch_encoder_only.input_ids.shape[1], 4)
                self.assertEqual(batch_encoder_only.attention_mask.shape[1], 4)
                self.assertNotIn('decoder_input_ids', batch_encoder_only)

    @unittest.skip('Unfortunately way too slow to build a BPE with SentencePiece.')
    def test_save_slow_from_fast_and_reload_fast(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_special_tokens_initialization(self):
        if False:
            return 10
        for (tokenizer, pretrained_name, kwargs) in self.tokenizers_list:
            with self.subTest(f'{tokenizer.__class__.__name__} ({pretrained_name})'):
                added_tokens = [AddedToken('<special>', lstrip=True)]
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, additional_special_tokens=added_tokens, **kwargs)
                r_output = tokenizer_r.encode('Hey this is a <special> token')
                special_token_id = tokenizer_r.encode('<special>', add_special_tokens=False)[0]
                self.assertTrue(special_token_id in r_output)
                if self.test_slow_tokenizer:
                    tokenizer_cr = self.rust_tokenizer_class.from_pretrained(pretrained_name, additional_special_tokens=added_tokens, **kwargs)
                    tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, additional_special_tokens=added_tokens, **kwargs)
                    p_output = tokenizer_p.encode('Hey this is a <special> token')
                    cr_output = tokenizer_cr.encode('Hey this is a <special> token')
                    self.assertEqual(p_output, r_output)
                    self.assertEqual(cr_output, r_output)
                    self.assertTrue(special_token_id in p_output)
                    self.assertTrue(special_token_id in cr_output)

    @unittest.skip('encode_plus and batch_encode_plus are deprecated and __call__ do some processing, so we expect different results.')
    def test_call(self):
        if False:
            print('Hello World!')
        pass

    def test_training_new_tokenizer(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.test_rust_tokenizer:
            return
        tokenizer = self.get_rust_tokenizer()
        new_tokenizer = tokenizer.train_new_from_iterator(SMALL_TRAINING_CORPUS, 100)
        inputs = new_tokenizer(['This is the first sentence', 'This sentence is different 🤗.'])
        self.assertEqual(len(inputs['input_ids']), 2)
        decoded_input = new_tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        expected_result = 'This is the first sentence'
        if tokenizer.backend_tokenizer.normalizer is not None:
            expected_result = tokenizer.backend_tokenizer.normalizer.normalize_str(expected_result)
        self.assertEqual(expected_result, decoded_input)
        new_tokenizer.tgt_lang = tokenizer.tgt_lang
        tokenizer.tgt_lang = tokenizer.tgt_lang
        self.assertEqual(tokenizer.num_special_tokens_to_add(False), new_tokenizer.num_special_tokens_to_add(False))
        self.assertEqual(tokenizer.num_special_tokens_to_add(True), new_tokenizer.num_special_tokens_to_add(True))
        self.assertEqual(tokenizer.max_len_single_sentence, new_tokenizer.max_len_single_sentence)
        self.assertEqual(tokenizer.max_len_sentences_pair, new_tokenizer.max_len_sentences_pair)
        self.assertSequenceEqual(tokenizer.all_special_tokens_extended, new_tokenizer.all_special_tokens_extended)
        self.assertDictEqual(tokenizer.special_tokens_map, new_tokenizer.special_tokens_map)

    @unittest.skip('Fails because of the hack of adding <unk> in _tokenize')
    def test_pickle_subword_regularization_tokenizer(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip('Fails because of the hack of adding <unk> in _tokenize')
    def test_subword_regularization_tokenizer(self):
        if False:
            i = 10
            return i + 15
        pass

@require_torch
@require_sentencepiece
@require_tokenizers
class SeamlessM4TDistilledIntegrationTest(unittest.TestCase):
    checkpoint_name = 'facebook/hf-seamless-m4t-medium'
    src_text = [' UN Chief Says There Is No Military Solution in Syria', ' Secretary-General Ban Ki-moon says his response to Russia\'s stepped up military support for Syria is that "there is no military solution" to the nearly five-year conflict and more weapons will only worsen the violence and misery for millions of people.']
    tgt_text = ['Şeful ONU declară că nu există o soluţie militară în Siria', 'Secretarul General Ban Ki-moon declară că răspunsul său la intensificarea sprijinului militar al Rusiei pentru Siria este că "nu există o soluţie militară" la conflictul de aproape cinci ani şi că noi arme nu vor face decât să înrăutăţească violenţele şi mizeria pentru milioane de oameni.']
    expected_src_tokens = [256047, 16297, 134408, 8165, 248066, 14734, 950, 1135, 105721, 3573, 83, 27352, 108, 49486, 3]

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        cls.tokenizer: SeamlessM4TTokenizer = SeamlessM4TTokenizer.from_pretrained(cls.checkpoint_name, src_lang='eng', tgt_lang='ron')
        return cls

    def test_language_codes(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.tokenizer.convert_tokens_to_ids('__ace_Latn__'), 256002)
        self.assertEqual(self.tokenizer.convert_tokens_to_ids('__shn__'), 256152)
        self.assertEqual(self.tokenizer.convert_tokens_to_ids('__eng__'), 256047)
        self.assertEqual(self.tokenizer.convert_tokens_to_ids('__fra__'), 256057)
        self.assertEqual(self.tokenizer.convert_tokens_to_ids('__quy__'), 256144)

    def test_tokenizer_tgt_lang(self):
        if False:
            while True:
                i = 10
        ids = self.tokenizer(self.src_text, src_lang='fra').input_ids[0]
        self.assertListEqual(self.expected_src_tokens[1:], ids[1:len(self.expected_src_tokens)])
        self.assertEqual(256057, ids[0])
        rest_ids = ids[len(self.expected_src_tokens):]
        self.assertListEqual([0] * len(rest_ids), rest_ids)
        ids = self.tokenizer(self.src_text, src_lang='__shn__').input_ids[0]
        self.assertListEqual(self.expected_src_tokens[1:], ids[1:len(self.expected_src_tokens)])
        self.assertEqual(256152, ids[0])

    def test_enro_tokenizer_decode_ignores_language_codes(self):
        if False:
            return 10
        self.assertIn(RO_CODE, self.tokenizer.all_special_ids)
        generated_ids = [RO_CODE, 4254, 98068, 112923, 39072, 3909, 713, 102767, 26, 17314, 35642, 14683, 33118, 2022, 66987, 2, 256047]
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        expected_romanian = self.tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
        self.assertEqual(result, expected_romanian)
        self.assertNotIn(self.tokenizer.eos_token, result)

    def test_enro_tokenizer_truncation(self):
        if False:
            return 10
        src_text = ['this is gunna be a long sentence ' * 20]
        assert isinstance(src_text[0], str)
        desired_max_length = 10
        ids = self.tokenizer(src_text, max_length=desired_max_length, truncation=True).input_ids[0]
        self.assertEqual(ids[-1], 3)
        self.assertEqual(ids[0], EN_CODE)
        self.assertEqual(len(ids), desired_max_length)

    def test_special_tokens_unaffacted_by_save_load(self):
        if False:
            print('Hello World!')
        tmpdirname = tempfile.mkdtemp()
        original_special_tokens = self.tokenizer.additional_special_tokens
        self.tokenizer.save_pretrained(tmpdirname)
        new_tok = SeamlessM4TTokenizer.from_pretrained(tmpdirname)
        self.assertListEqual(new_tok.additional_special_tokens, original_special_tokens)

    @require_torch
    def test_enro_tokenizer_prepare_batch(self):
        if False:
            i = 10
            return i + 15
        batch = self.tokenizer(self.src_text, text_target=self.tgt_text, padding=True, truncation=True, max_length=len(self.expected_src_tokens), pad_to_multiple_of=None, return_tensors='pt')
        batch['decoder_input_ids'] = shift_tokens_right(batch['labels'], self.tokenizer.pad_token_id, self.tokenizer.convert_tokens_to_ids('__ron__'))
        self.assertIsInstance(batch, BatchEncoding)
        self.assertEqual((2, 15), batch.input_ids.shape)
        self.assertEqual((2, 15), batch.attention_mask.shape)
        result = batch.input_ids.tolist()[0]
        self.assertListEqual(self.expected_src_tokens, result)
        self.assertEqual(RO_CODE, batch.decoder_input_ids[0, 0])
        self.assertEqual(self.tokenizer.prefix_tokens, [EN_CODE])
        self.assertEqual(self.tokenizer.suffix_tokens, [self.tokenizer.eos_token_id])

    def test_seq2seq_max_length(self):
        if False:
            while True:
                i = 10
        batch = self.tokenizer(self.src_text, padding=True, truncation=True, max_length=3, return_tensors='pt', pad_to_multiple_of=None)
        targets = self.tokenizer(text_target=self.tgt_text, padding=True, truncation=True, max_length=10, return_tensors='pt')
        labels = targets['input_ids']
        batch['decoder_input_ids'] = shift_tokens_right(labels, self.tokenizer.pad_token_id, decoder_start_token_id=self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang))
        self.assertEqual(batch.input_ids.shape[1], 3)
        self.assertEqual(batch.decoder_input_ids.shape[1], 10)

    @require_torch
    def test_tokenizer_translation(self):
        if False:
            while True:
                i = 10
        inputs = self.tokenizer._build_translation_inputs('A test', return_tensors='pt', src_lang='eng', tgt_lang='fra')
        self.assertEqual(nested_simplify(inputs), {'input_ids': [[256047, 70, 7356, 3]], 'attention_mask': [[1, 1, 1, 1]], 'forced_bos_token_id': 256057})

@require_sentencepiece
@require_tokenizers
class CommonSpmIntegrationTests(unittest.TestCase):
    """
    A class that regroups important test to make sure that we properly handle the special tokens.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        tokenizer = SeamlessM4TTokenizer(SAMPLE_VOCAB, extra_ids=0, add_bos_token=False, legacy=False)
        tokenizer.add_special_tokens({'additional_special_tokens': [AddedToken('<s>', rstrip=False, lstrip=False)]})
        cls.tokenizer = tokenizer
        return cls

    def test_add_dummy_prefix(self):
        if False:
            i = 10
            return i + 15
        input_ids = self.tokenizer.encode('. Hello')
        self.assertEqual(input_ids, [3, 1, 8, 5, 157, 87, 21, 3])
        sp_encode = self.tokenizer.sp_model.encode('. Hello')
        self.assertEqual(input_ids[:-1], [3, 1, 8] + [i + self.tokenizer.fairseq_offset for i in sp_encode])
        tokens = self.tokenizer.tokenize('. Hello')
        self.assertEqual(tokens, ['▁', '.', '▁He', 'll', 'o'])
        tokens = self.tokenizer.tokenize('')
        self.assertEqual(tokens, [])
        self.assertEqual(tokens, self.tokenizer.sp_model.encode('', out_type=str))
        tokens = self.tokenizer.tokenize(' ')
        self.assertEqual(tokens, [])
        self.assertEqual(tokens, self.tokenizer.sp_model.encode(' ', out_type=str))
        tokens = self.tokenizer.tokenize('▁')
        self.assertEqual(tokens, [])
        self.assertEqual(tokens, self.tokenizer.sp_model.encode('▁', out_type=str))

    def test_remove_extra_whitespaces(self):
        if False:
            for i in range(10):
                print('nop')
        input_ids = self.tokenizer.encode('       . Hello')
        self.assertEqual(input_ids, [3, 1, 8, 5, 157, 87, 21, 3])
        sp_encode = self.tokenizer.sp_model.encode('       . Hello')
        self.assertEqual([i - self.tokenizer.fairseq_offset for i in input_ids[2:-1]], [7] + sp_encode)
        tokens = self.tokenizer.tokenize(' . Hello')
        self.assertEqual(tokens, ['▁', '.', '▁He', 'll', 'o'])
        input_ids = self.tokenizer.encode('▁He is not')
        self.assertEqual(input_ids, [3, 1, 157, 47, 45, 3])
        tokens = self.tokenizer.tokenize('▁He is not')
        sp_encode = [self.tokenizer.sp_model.piece_to_id('▁He'), self.tokenizer.sp_model.piece_to_id('▁is'), self.tokenizer.sp_model.piece_to_id('▁not')]
        self.assertEqual([i - self.tokenizer.fairseq_offset for i in input_ids[2:-1]], sp_encode)
        self.assertEqual(tokens, ['▁He', '▁is', '▁not'])
        input_ids = self.tokenizer.encode('▁He is not<s>             ▁He')
        self.assertEqual(input_ids, [3, 1, 157, 47, 45, 2, 157, 3])
        tokens = self.tokenizer.tokenize('▁He is not<s>              ▁He')
        self.assertEqual(tokens, ['▁He', '▁is', '▁not', '<s>', '▁He'])
        input_ids = self.tokenizer.encode('▁He is not             ▁He')
        self.assertEqual(input_ids, [3, 1, 157, 47, 45, 157, 3])
        tokens = self.tokenizer.tokenize('▁He is not              ▁He')
        self.assertEqual(tokens, ['▁He', '▁is', '▁not', '▁He'])

    def test_character_after_special_token(self):
        if False:
            for i in range(10):
                print('nop')
        input_ids = self.tokenizer.encode('Hey <s>I')
        self.assertEqual(input_ids, [3, 1, 157, 31, 2, 101, 3])
        sp_encode = self.tokenizer.sp_model.encode('Hey .I')
        self.assertEqual(input_ids[-2] - self.tokenizer.fairseq_offset, sp_encode[-1])
        tokens = self.tokenizer.tokenize('<s>I')
        self.assertEqual(tokens, ['<s>', 'I'])
        input_ids = self.tokenizer.encode('Hello, <s>,')
        self.assertEqual(input_ids, [3, 1, 157, 87, 21, 4, 2, 4, 3])
        tokens = self.tokenizer.tokenize('Hello, <s>,')
        self.assertEqual(tokens, ['▁He', 'll', 'o', ',', '<s>', ','])

    def test_special_tokens_strip(self):
        if False:
            i = 10
            return i + 15
        input_ids = self.tokenizer.encode(' <s> ,')
        self.assertEqual(input_ids, [3, 1, 2, 8, 4, 3])
        tokens = self.tokenizer.tokenize(' <s> ,')
        self.assertEqual(tokens, ['<s>', '▁', ','])
        input_ids = self.tokenizer.encode('No <s> ▁He')
        self.assertEqual(input_ids, [3, 1, 285, 2, 157, 3])
        tokens = self.tokenizer.tokenize('No <s> ▁He')
        self.assertEqual(tokens, ['▁No', '<s>', '▁He'])