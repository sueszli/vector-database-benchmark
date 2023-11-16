import os
import unittest
from os import path as osp
import cv2
from PIL import Image
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import created_boxed_image
from modelscope.utils.test_utils import test_level

class OfaTasksTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.output_dir = 'unittest_output'
        os.makedirs(self.output_dir, exist_ok=True)

    def save_img(self, image_in, box, image_out):
        if False:
            while True:
                i = 10
        cv2.imwrite(osp.join(self.output_dir, image_out), created_boxed_image(image_in, box))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_image_captioning_with_model(self):
        if False:
            i = 10
            return i + 15
        model = Model.from_pretrained('damo/ofa_image-caption_coco_large_en')
        img_captioning = pipeline(task=Tasks.image_captioning, model=model)
        image = 'data/test/images/image_captioning.png'
        result = img_captioning(image)
        print(result[OutputKeys.CAPTION])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_image_captioning_with_name(self):
        if False:
            while True:
                i = 10
        img_captioning = pipeline(Tasks.image_captioning, model='damo/ofa_image-caption_coco_large_en')
        result = img_captioning('data/test/images/image_captioning.png')
        print(result[OutputKeys.CAPTION])
        img_captioning.model.num_return_sequences = 2
        result = img_captioning('data/test/images/image_captioning.png')
        print(result[OutputKeys.CAPTION])
        img_captioning.model.num_return_sequences = 1
        results = img_captioning([{'image': 'data/test/images/image_captioning.png'} for _ in range(3)], batch_size=2)
        for r in results:
            print(r[OutputKeys.CAPTION])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_ocr_recognize_with_name(self):
        if False:
            return 10
        ocr_recognize = pipeline(Tasks.ocr_recognition, model='damo/ofa_ocr-recognition_scene_base_zh')
        result = ocr_recognize('data/test/images/image_ocr_recognition.jpg')
        print(result[OutputKeys.TEXT])
        results = ocr_recognize(['data/test/images/image_ocr_recognition.jpg' for _ in range(3)], batch_size=2)
        for r in results:
            print(r[OutputKeys.TEXT])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_image_classification_with_model(self):
        if False:
            print('Hello World!')
        model = Model.from_pretrained('damo/ofa_image-classification_imagenet_large_en')
        ofa_pipe = pipeline(Tasks.image_classification, model=model)
        image = 'data/test/images/image_classification.png'
        result = ofa_pipe(image)
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_image_classification_with_name(self):
        if False:
            for i in range(10):
                print('nop')
        ofa_pipe = pipeline(Tasks.image_classification, model='damo/ofa_image-classification_imagenet_large_en')
        image = 'data/test/images/image_classification.png'
        result = ofa_pipe(image)
        print(result)
        image = ['data/test/images/image_classification.png' for _ in range(3)]
        results = ofa_pipe(image, batch_size=2)
        for r in results:
            print(r[OutputKeys.LABELS], r[OutputKeys.SCORES])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_summarization_with_model(self):
        if False:
            return 10
        model = Model.from_pretrained('damo/ofa_summarization_gigaword_large_en')
        ofa_pipe = pipeline(Tasks.text_summarization, model=model)
        text = 'five-time world champion michelle kwan withdrew' + 'from the #### us figure skating championships on wednesday ,' + ' but will petition us skating officials for the chance to ' + 'compete at the #### turin olympics .'
        input = {'text': text}
        result = ofa_pipe(input)
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_summarization_with_name(self):
        if False:
            return 10
        ofa_pipe = pipeline(Tasks.text_summarization, model='damo/ofa_summarization_gigaword_large_en')
        text = 'five-time world champion michelle kwan withdrew' + 'from the #### us figure skating championships on wednesday ,' + ' but will petition us skating officials for the chance to ' + 'compete at the #### turin olympics .'
        input = {'text': text}
        result = ofa_pipe(input)
        print(result)
        ofa_pipe.model.num_return_sequences = 2
        result = ofa_pipe(input)
        print(result)
        ofa_pipe.model.num_return_sequences = 1
        input = [{'text': text} for _ in range(3)]
        results = ofa_pipe(input, batch_size=2)
        for r in results:
            print(r[OutputKeys.TEXT])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_text_classification_with_model(self):
        if False:
            for i in range(10):
                print('nop')
        model = Model.from_pretrained('damo/ofa_text-classification_mnli_large_en')
        ofa_pipe = pipeline(Tasks.text_classification, model=model)
        text = 'One of our number will carry out your instructions minutely.'
        text2 = 'A member of my team will execute your orders with immense precision.'
        result = ofa_pipe((text, text2))
        result = ofa_pipe({'text': text, 'text2': text2})
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_text_classification_with_name(self):
        if False:
            print('Hello World!')
        ofa_pipe = pipeline(Tasks.text_classification, model='damo/ofa_text-classification_mnli_large_en')
        text = 'One of our number will carry out your instructions minutely.'
        text2 = 'A member of my team will execute your orders with immense precision.'
        result = ofa_pipe((text, text2))
        print(result)
        inputs = [(text, text2) for _ in range(3)]
        results = ofa_pipe(inputs, batch_size=2)
        for r in results:
            print(r[OutputKeys.LABELS], r[OutputKeys.SCORES])

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_visual_entailment_with_model(self):
        if False:
            print('Hello World!')
        model = Model.from_pretrained('damo/ofa_visual-entailment_snli-ve_large_en')
        ofa_pipe = pipeline(Tasks.visual_entailment, model=model)
        image = 'data/test/images/dogs.jpg'
        text = 'there are two birds.'
        input = {'image': image, 'text': text}
        result = ofa_pipe(input)
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_visual_entailment_with_name(self):
        if False:
            i = 10
            return i + 15
        ofa_pipe = pipeline(Tasks.visual_entailment, model='damo/ofa_visual-entailment_snli-ve_large_en')
        image = 'data/test/images/dogs.jpg'
        text = 'there are two birds.'
        input = {'image': image, 'text': text}
        result = ofa_pipe(input)
        print(result)
        input = [{'image': image, 'text': text} for _ in range(3)]
        results = ofa_pipe(input, batch_size=2)
        for r in results:
            print(r[OutputKeys.LABELS], r[OutputKeys.SCORES])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_visual_grounding_with_model(self):
        if False:
            while True:
                i = 10
        model = Model.from_pretrained('damo/ofa_visual-grounding_refcoco_large_en')
        ofa_pipe = pipeline(Tasks.visual_grounding, model=model)
        image = 'data/test/images/visual_grounding.png'
        text = 'a blue turtle-like pokemon with round head'
        input = {'image': image, 'text': text}
        result = ofa_pipe(input)
        print(result)
        image_name = image.split('/')[-2]
        self.save_img(image, result[OutputKeys.BOXES][0], osp.join('large_en_model_' + image_name + '.png'))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_visual_grounding_with_name(self):
        if False:
            print('Hello World!')
        ofa_pipe = pipeline(Tasks.visual_grounding, model='damo/ofa_visual-grounding_refcoco_large_en')
        image = 'data/test/images/visual_grounding.png'
        text = 'a blue turtle-like pokemon with round head'
        input = {'image': image, 'text': text}
        result = ofa_pipe(input)
        print(result)
        image_name = image.split('/')[-2]
        self.save_img(image, result[OutputKeys.BOXES][0], osp.join('large_en_name_' + image_name + '.png'))
        result = ofa_pipe([input for _ in range(3)], batch_size=2)
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_visual_grounding_zh_with_name(self):
        if False:
            while True:
                i = 10
        model = 'damo/ofa_visual-grounding_refcoco_large_zh'
        ofa_pipe = pipeline(Tasks.visual_grounding, model=model)
        image = 'data/test/images/visual_grounding.png'
        text = '一个圆头的蓝色宝可梦'
        input = {'image': image, 'text': text}
        result = ofa_pipe(input)
        print(result)
        image_name = image.split('/')[-1]
        self.save_img(image, result[OutputKeys.BOXES][0], osp.join('large_zh_name_' + image_name))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_visual_question_answering_with_model(self):
        if False:
            i = 10
            return i + 15
        model = Model.from_pretrained('damo/ofa_visual-question-answering_pretrain_large_en')
        ofa_pipe = pipeline(Tasks.visual_question_answering, model=model)
        image = 'data/test/images/visual_question_answering.png'
        text = 'what is grown on the plant?'
        input = {'image': image, 'text': text}
        result = ofa_pipe(input)
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_visual_question_answering_with_name(self):
        if False:
            for i in range(10):
                print('nop')
        model = 'damo/ofa_visual-question-answering_pretrain_large_en'
        ofa_pipe = pipeline(Tasks.visual_question_answering, model=model)
        image = 'data/test/images/visual_question_answering.png'
        text = 'what is grown on the plant?'
        input = {'image': image, 'text': text}
        result = ofa_pipe(input)
        print(result)
        result = ofa_pipe([input for _ in range(3)], batch_size=2)
        print(result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_image_captioning_distilled_with_model(self):
        if False:
            i = 10
            return i + 15
        model = Model.from_pretrained('damo/ofa_image-caption_coco_distilled_en')
        img_captioning = pipeline(task=Tasks.image_captioning, model=model)
        image_path = 'data/test/images/image_captioning.png'
        image = Image.open(image_path)
        result = img_captioning(image)
        print(result[OutputKeys.CAPTION])
        print(img_captioning([image for _ in range(3)], batch_size=2))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_visual_entailment_distilled_model_with_name(self):
        if False:
            print('Hello World!')
        ofa_pipe = pipeline(Tasks.visual_entailment, model='damo/ofa_visual-entailment_snli-ve_distilled_v2_en')
        image = 'data/test/images/dogs.jpg'
        text = 'there are two birds.'
        input = {'image': image, 'text': text}
        result = ofa_pipe(input)
        print(result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_visual_grounding_distilled_model_with_model(self):
        if False:
            return 10
        model = Model.from_pretrained('damo/ofa_visual-grounding_refcoco_distilled_en')
        ofa_pipe = pipeline(Tasks.visual_grounding, model=model)
        image = 'data/test/images/visual_grounding.png'
        text = 'a blue turtle-like pokemon with round head'
        input = {'image': image, 'text': text}
        result = ofa_pipe(input)
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_text_to_image_synthesis_with_name(self):
        if False:
            while True:
                i = 10
        model = 'damo/ofa_text-to-image-synthesis_coco_large_en'
        ofa_pipe = pipeline(Tasks.text_to_image_synthesis, model=model)
        ofa_pipe.model.generator.beam_size = 2
        example = {'text': 'a bear in the water.'}
        result = ofa_pipe(example)
        result[OutputKeys.OUTPUT_IMGS][0].save('result.png')
        print(f"Output written to {osp.abspath('result.png')}")

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_text_to_image_synthesis_with_model(self):
        if False:
            for i in range(10):
                print('nop')
        model = Model.from_pretrained('damo/ofa_text-to-image-synthesis_coco_large_en')
        ofa_pipe = pipeline(Tasks.text_to_image_synthesis, model=model)
        ofa_pipe.model.generator.beam_size = 2
        example = {'text': 'a bear in the water.'}
        result = ofa_pipe(example)
        result[OutputKeys.OUTPUT_IMG].save('result.png')
        print(f"Output written to {osp.abspath('result.png')}")

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_asr_with_name(self):
        if False:
            return 10
        model = 'damo/ofa_mmspeech_pretrain_base_zh'
        ofa_pipe = pipeline(Tasks.auto_speech_recognition, model=model)
        example = {'wav': 'data/test/audios/asr_example_ofa.wav'}
        result = ofa_pipe(example)
        print(result[OutputKeys.TEXT])
        result = ofa_pipe([example for _ in range(3)], batch_size=2)
        for r in result:
            print(r[OutputKeys.TEXT])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_sudoku_with_name(self):
        if False:
            for i in range(10):
                print('nop')
        model = 'damo/ofa_sudoku_kaggle_large'
        ofa_pipe = pipeline(Tasks.sudoku, model=model)
        example = '5 : 3 : 0 : 0 : 7 : 0 : 0 : 0 : 0 |                 6 : 0 : 0 : 1 : 9 : 5 : 0 : 0 : 0 |                 0 : 9 : 8 : 0 : 0 : 0 : 0 : 6 : 0 |                 8 : 0 : 0 : 0 : 6 : 0 : 0 : 0 : 3 |                 4 : 0 : 0 : 8 : 0 : 3 : 0 : 0 : 1 |                 7 : 0 : 0 : 0 : 2 : 0 : 0 : 0 : 6 |                 0 : 6 : 0 : 0 : 0 : 0 : 2 : 8 : 0 |                 0 : 0 : 0 : 4 : 1 : 9 : 0 : 0 : 5 |                 0 : 0 : 0 : 0 : 8 : 0 : 0 : 7 : 9'
        result = ofa_pipe(example)
        print(result[OutputKeys.TEXT])
        result = ofa_pipe([example for _ in range(3)], batch_size=2)
        for r in result:
            print(r[OutputKeys.TEXT])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_text2sql_with_name(self):
        if False:
            return 10
        model = 'damo/ofa_text2sql_spider_large_en'
        ofa_pipe = pipeline(Tasks.text2sql, model=model)
        text = 'Show all book categories and the number of books in each category.'
        database = 'culture_company'
        example = {'text': text, 'database': database}
        result = ofa_pipe(example)
        print(result[OutputKeys.TEXT])
        result = ofa_pipe([example for _ in range(3)], batch_size=2)
        for r in result:
            print(r[OutputKeys.TEXT])
if __name__ == '__main__':
    unittest.main()