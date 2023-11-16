import unittest
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

class Image2ImageTranslationTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub(self):
        if False:
            print('Hello World!')
        "We provide three translation modes, i.e., uncropping, colorization and combination.\n            You can pass the following parameters for different mode.\n            1. Uncropping Mode:\n            result = img2img_gen_pipeline(('data/test/images/img2img_input.jpg', 'left', 0, 'result.jpg'))\n            2. Colorization Mode:\n            result = img2img_gen_pipeline(('data/test/images/img2img_input.jpg', 1, 'result.jpg'))\n            3. Combination Mode:\n            just like the following code.\n        "
        img2img_gen_pipeline = pipeline(Tasks.image_to_image_translation, model='damo/cv_latent_diffusion_image2image_translation')
        result = img2img_gen_pipeline(('data/test/images/img2img_input_mask.png', 'data/test/images/img2img_input_masked_img.png', 2, 'result.jpg'))
        print(f'output: {result}.')
if __name__ == '__main__':
    unittest.main()