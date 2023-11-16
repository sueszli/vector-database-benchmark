import os
import shutil
import tempfile
import unittest
import numpy as np
import pytest
import requests
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers, require_torch, require_vision
from transformers.utils import is_vision_available
if is_vision_available():
    from PIL import Image
    from transformers import AutoProcessor, CLIPImageProcessor, Kosmos2Processor, PreTrainedTokenizerFast, XLMRobertaTokenizer, XLMRobertaTokenizerFast
SAMPLE_VOCAB = get_tests_dir('fixtures/test_sentencepiece.model')

@require_sentencepiece
@require_tokenizers
@require_vision
class Kosmos2ProcessorTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.tmpdirname = tempfile.mkdtemp()
        image_processor = CLIPImageProcessor(use_square_size=True)
        slow_tokenizer = XLMRobertaTokenizer(SAMPLE_VOCAB)
        fast_tokenizer = XLMRobertaTokenizerFast(__slow_tokenizer=slow_tokenizer)
        processor = Kosmos2Processor(image_processor, fast_tokenizer)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        if False:
            return 10
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        if False:
            print('Hello World!')
        shutil.rmtree(self.tmpdirname)

    def prepare_image_inputs(self):
        if False:
            return 10
        'This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,\n        or a list of PyTorch tensors if one specifies torchify=True.\n        '
        image_inputs = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]
        image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]
        return image_inputs

    def test_save_load_pretrained_additional_features(self):
        if False:
            print('Hello World!')
        processor = Kosmos2Processor(tokenizer=self.get_tokenizer(), image_processor=self.get_image_processor())
        processor.save_pretrained(self.tmpdirname)
        tokenizer_add_kwargs = self.get_tokenizer(bos_token='(BOS)', eos_token='(EOS)')
        image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)
        processor = Kosmos2Processor.from_pretrained(self.tmpdirname, bos_token='(BOS)', eos_token='(EOS)', do_normalize=False, padding_value=1.0)
        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, PreTrainedTokenizerFast)
        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, CLIPImageProcessor)

    def test_image_processor(self):
        if False:
            i = 10
            return i + 15
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        processor = Kosmos2Processor(tokenizer=tokenizer, image_processor=image_processor)
        image_input = self.prepare_image_inputs()
        input_image_processor = image_processor(image_input, return_tensors='np')
        input_processor = processor(images=image_input, return_tensors='np')
        for key in input_image_processor.keys():
            self.assertAlmostEqual(input_image_processor[key].sum(), input_processor[key].sum(), delta=0.01)

    def test_tokenizer(self):
        if False:
            return 10
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        processor = Kosmos2Processor(tokenizer=tokenizer, image_processor=image_processor)
        input_str = 'This is a test'
        encoded_processor = processor(text=input_str, add_eos_token=True)
        encoded_tok = tokenizer(input_str, return_token_type_ids=False)
        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_processor(self):
        if False:
            while True:
                i = 10
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        processor = Kosmos2Processor(tokenizer=tokenizer, image_processor=image_processor)
        input_str = 'This is a test'
        image_input = self.prepare_image_inputs()
        inputs = processor(text=input_str, images=image_input)
        self.assertListEqual(list(inputs.keys()), ['pixel_values', 'input_ids', 'attention_mask', 'image_embeds_position_mask'])
        with pytest.raises(ValueError):
            processor()

    def test_tokenizer_decode(self):
        if False:
            for i in range(10):
                print('nop')
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        processor = Kosmos2Processor(tokenizer=tokenizer, image_processor=image_processor)
        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]
        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)
        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        if False:
            return 10
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        processor = Kosmos2Processor(tokenizer=tokenizer, image_processor=image_processor)
        input_str = 'This is a test'
        image_input = self.prepare_image_inputs()
        inputs = processor(text=input_str, images=image_input)
        self.assertListEqual(list(inputs.keys()), ['pixel_values', 'input_ids', 'attention_mask', 'image_embeds_position_mask'])
        inputs = processor(text=input_str)
        self.assertListEqual(list(inputs.keys()), ['input_ids', 'attention_mask'])
        inputs = processor(images=image_input)
        self.assertListEqual(list(inputs.keys()), ['pixel_values'])

    @require_torch
    def test_full_processor(self):
        if False:
            return 10
        url = 'https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/two_dogs.jpg'
        processor = Kosmos2Processor.from_pretrained('microsoft/kosmos-2-patch14-224')
        texts = ['<grounding> Two puppies sit in a field of grass.', '<grounding> <phrase> Two puppies </phrase> sit in a field of grass.', '<grounding> <phrase> Two puppies </phrase> sit in a field of <phrase> grass </phrase>.', '<grounding> <phrase> Two puppies </phrase> <object> <patch_index_0079> <patch_index_1016> </delimiter_of_multi_objects/> <patch_index_0135> <patch_index_1008> </object> sit in a field of <phrase> grass </phrase>.']
        image = Image.open(requests.get(url, stream=True).raw)
        image_path = os.path.join(self.tmpdirname, 'image.jpg')
        image.save(image_path)
        image = Image.open(image_path)
        bboxes = [[None, []], [[None], [[]], [(79, 1016)], [[(79, 1016)]], [[(79, 1016), (135, 1008)]]], [[[(79, 1016), (135, 1008)], None], [[(79, 1016), (135, 1008)], []], [[(79, 1016), (135, 1008)], (480, 1023)], [[(79, 1016), (135, 1008)], [(480, 1023)]]], [[None, [(480, 1023)]]]]
        batch_image = [image] * 4
        batch_text = [texts[0], texts[1], texts[1], texts[2]]
        batch_bboxes = [None, [[]], [(79, 1016)], [[(79, 1016), (135, 1008)], (480, 1023)]]
        expected_input_ids = [[0, 64012, 1264, 17772, 1357, 12, 10, 770, 9, 4464, 4, 2], [0, 64012, 64007, 1264, 17772, 64008, 1357, 12, 10, 770, 9, 4464, 4, 2], [0, 64012, 64007, 1264, 17772, 64008, 64009, 64092, 65029, 64010, 1357, 12, 10, 770, 9, 4464, 4, 2], [0, 64012, 64007, 1264, 17772, 64008, 64009, 64092, 65029, 64011, 64148, 65021, 64010, 1357, 12, 10, 770, 9, 4464, 4, 2], [0, 64012, 64007, 1264, 17772, 64008, 64009, 64092, 65029, 64011, 64148, 65021, 64010, 1357, 12, 10, 770, 9, 64007, 4464, 64008, 106, 4, 2], [0, 64012, 64007, 1264, 17772, 64008, 64009, 64092, 65029, 64011, 64148, 65021, 64010, 1357, 12, 10, 770, 9, 64007, 4464, 64008, 64009, 64493, 65036, 64010, 106, 4, 2]]
        EXPECTED_PIXEL_VALUES_1 = np.array([[[-0.6535852551460266, -0.6389868259429932, -0.6243883967399597], [-0.6535852551460266, -0.6389868259429932, -0.6243883967399597], [-0.6243883967399597, -0.6243883967399597, -0.5951915383338928]], [[-0.20629698038101196, -0.19128920137882233, -0.19128920137882233], [-0.20629698038101196, -0.19128920137882233, -0.17628143727779388], [-0.2213047444820404, -0.20629698038101196, -0.16127367317676544]], [[-0.5843556523323059, -0.5701355338096619, -0.5701355338096619], [-0.5843556523323059, -0.5701355338096619, -0.5559154152870178], [-0.5843556523323059, -0.5559154152870178, -0.5416953563690186]]])
        EXPECTED_PIXEL_VALUES_2 = np.array([[[-0.4346088469028473, -0.47840413451194763, -0.7849710583686829], [-0.5221993923187256, -0.5076009631156921, -0.755774199962616], [-0.5221993923187256, -0.5076009631156921, -0.7411757707595825]], [[-0.2813358008861542, -0.2963435649871826, -0.431413471698761], [-0.26632803678512573, -0.2963435649871826, -0.4764367938041687], [-0.2213047444820404, -0.2813358008861542, -0.49144455790519714]], [[-0.5701355338096619, -0.641235888004303, -0.7549964189529419], [-0.5843556523323059, -0.641235888004303, -0.7834365367889404], [-0.5559154152870178, -0.641235888004303, -0.7834365367889404]]])

        def check(texts, bboxes, expected_input_ids):
            if False:
                return 10
            outputs = processor(images=None, text=texts, bboxes=bboxes, add_eos_token=True)
            self.assertListEqual(outputs.input_ids, expected_input_ids)
        check(texts[0], bboxes[0][0], expected_input_ids[0])
        check(texts[0], bboxes[0][1], expected_input_ids[0])
        check(texts[1], bboxes[1][0], expected_input_ids[1])
        check(texts[1], bboxes[1][1], expected_input_ids[1])
        check(texts[1], bboxes[1][2], expected_input_ids[2])
        check(texts[1], bboxes[1][3], expected_input_ids[2])
        check(texts[1], bboxes[1][4], expected_input_ids[3])
        with pytest.raises(ValueError):
            _ = processor.preprocess_examples(images=None, texts=texts[1], bboxes=[[None]])
        check(texts[2], bboxes[2][0], expected_input_ids[4])
        check(texts[2], bboxes[2][1], expected_input_ids[4])
        check(texts[2], bboxes[2][2], expected_input_ids[5])
        check(texts[2], bboxes[2][3], expected_input_ids[5])
        check(texts[3], bboxes[3][0], expected_input_ids[5])
        with pytest.raises(ValueError):
            _ = processor.preprocess_examples(images=None, texts=texts[2], bboxes=[[(79, 1016), (135, 1008)], [None]])
        outputs = processor(images=None, text=batch_text, bboxes=batch_bboxes, add_eos_token=True)
        self.assertListEqual(outputs.input_ids, [expected_input_ids[0], expected_input_ids[1], expected_input_ids[2], expected_input_ids[5]])
        outputs = processor(images=None, text=batch_text, bboxes=batch_bboxes, padding=True, add_eos_token=True)
        self.assertListEqual(outputs.input_ids[0], expected_input_ids[0] + [1] * (len(expected_input_ids[5]) - len(expected_input_ids[0])))
        self.assertListEqual(outputs.attention_mask[0], [1] * len(expected_input_ids[0]) + [0] * (len(expected_input_ids[5]) - len(expected_input_ids[0])))
        self.assertListEqual(outputs.input_ids[-1], expected_input_ids[5])
        self.assertListEqual(outputs.attention_mask[-1], [1] * len(expected_input_ids[5]))
        outputs = processor(images=None, text=batch_text, bboxes=batch_bboxes, return_tensors='pt', padding=True, add_eos_token=True)
        self.assertListEqual(outputs.input_ids.numpy().tolist()[0], expected_input_ids[0] + [1] * (len(expected_input_ids[5]) - len(expected_input_ids[0])))
        self.assertListEqual(outputs.attention_mask.numpy().tolist()[0], [1] * len(expected_input_ids[0]) + [0] * (len(expected_input_ids[5]) - len(expected_input_ids[0])))
        self.assertListEqual(outputs.input_ids.numpy().tolist()[-1], expected_input_ids[5])
        self.assertListEqual(outputs.attention_mask.numpy().tolist()[-1], [1] * len(expected_input_ids[5]))
        num_image_tokens = 64
        outputs = processor(images=image, text=texts[0], bboxes=None, add_eos_token=True)
        self.assertTupleEqual(outputs.pixel_values[0].shape, (3, 224, 224))
        self.assertListEqual(outputs.input_ids, [0, 64003] + list(range(4, 4 + num_image_tokens)) + [64004] + expected_input_ids[0][1:])
        self.assertListEqual(outputs.image_embeds_position_mask, [0] * 2 + [1] * num_image_tokens + [0] + [0] * (len(expected_input_ids[0]) - 1))
        np.testing.assert_allclose(outputs.pixel_values[0][:3, :3, :3], EXPECTED_PIXEL_VALUES_1, atol=1e-09)
        np.testing.assert_allclose(outputs.pixel_values[0][:3, -3:, -3:], EXPECTED_PIXEL_VALUES_2, atol=1e-09)
        outputs = processor(images=batch_image, text=batch_text, bboxes=batch_bboxes, return_tensors='pt', padding=True, add_eos_token=True)
        self.assertTupleEqual(outputs.pixel_values.shape, (4, 3, 224, 224))
        np.testing.assert_allclose(outputs.pixel_values[:, :3, :3, :3].numpy(), [EXPECTED_PIXEL_VALUES_1] * len(batch_image), atol=1e-09)
        np.testing.assert_allclose(outputs.pixel_values[:, :3, -3:, -3:].numpy(), [EXPECTED_PIXEL_VALUES_2] * len(batch_image), atol=1e-09)
        EXPECTED_IDS_BATCH_RIGHT_PADDING = [[0, 64003] + list(range(4, 4 + num_image_tokens)) + [64004] + expected_input_ids[0][1:] + [1] * (len(expected_input_ids[5]) - len(expected_input_ids[0])), [0, 64003] + list(range(4, 4 + num_image_tokens)) + [64004] + expected_input_ids[5][1:]]
        EXPECTED_MASK_BATCH_RIGHT_PADDING = [[1, 1] + [1] * num_image_tokens + [1] + [1] * len(expected_input_ids[0][1:]) + [0] * (len(expected_input_ids[5]) - len(expected_input_ids[0])), [1] * (2 + num_image_tokens + len(expected_input_ids[5]))]
        self.assertListEqual(outputs.input_ids.numpy().tolist()[0], EXPECTED_IDS_BATCH_RIGHT_PADDING[0])
        self.assertListEqual(outputs.attention_mask.numpy().tolist()[0], EXPECTED_MASK_BATCH_RIGHT_PADDING[0])
        self.assertListEqual(outputs.input_ids.numpy().tolist()[-1], EXPECTED_IDS_BATCH_RIGHT_PADDING[-1])
        self.assertListEqual(outputs.attention_mask.numpy().tolist()[-1], EXPECTED_MASK_BATCH_RIGHT_PADDING[-1])
        self.assertListEqual(outputs.image_embeds_position_mask.numpy().tolist(), [[0, 0] + [1] * num_image_tokens + [0] + [0] * (len(expected_input_ids[5]) - 1)] * len(batch_image))
        processor = Kosmos2Processor.from_pretrained('microsoft/kosmos-2-patch14-224', padding_side='left')
        outputs = processor(images=batch_image, text=batch_text, bboxes=batch_bboxes, return_tensors='pt', padding=True, add_eos_token=True)
        EXPECTED_IDS_BATCH = [[1] * (len(expected_input_ids[5]) - len(expected_input_ids[0])) + [0, 64003] + list(range(4, 4 + num_image_tokens)) + [64004] + expected_input_ids[0][1:], [0, 64003] + list(range(4, 4 + num_image_tokens)) + [64004] + expected_input_ids[5][1:]]
        EXPECTED_MASK_BATCH = [[0] * (len(expected_input_ids[5]) - len(expected_input_ids[0])) + [1, 1] + [1] * num_image_tokens + [1] + [1] * len(expected_input_ids[0][1:]), [1] * (2 + num_image_tokens + len(expected_input_ids[5]))]
        EXPECTED_IMG_POS_MASK_BATCH = [[0] * (len(expected_input_ids[5]) - len(expected_input_ids[0])) + [0, 0] + [1] * num_image_tokens + [0] + [0] * len(expected_input_ids[0][1:]), [0, 0] + [1] * num_image_tokens + [0] + [0] * (len(expected_input_ids[5]) - 1)]
        self.assertListEqual(outputs.input_ids.numpy().tolist()[0], EXPECTED_IDS_BATCH[0])
        self.assertListEqual(outputs.attention_mask.numpy().tolist()[0], EXPECTED_MASK_BATCH[0])
        self.assertListEqual(outputs.image_embeds_position_mask.numpy().tolist()[0], EXPECTED_IMG_POS_MASK_BATCH[0])
        self.assertListEqual(outputs.input_ids.numpy().tolist()[-1], EXPECTED_IDS_BATCH[-1])
        self.assertListEqual(outputs.attention_mask.numpy().tolist()[-1], EXPECTED_MASK_BATCH[-1])
        self.assertListEqual(outputs.image_embeds_position_mask.numpy().tolist()[-1], EXPECTED_IMG_POS_MASK_BATCH[-1])