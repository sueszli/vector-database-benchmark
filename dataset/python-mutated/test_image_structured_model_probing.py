import unittest
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

class ImageStructuredModelProbingTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.task = Tasks.image_classification
        self.model_id = 'damo/structured_model_probing'

    @unittest.skip('skip test due to model is private')
    def test_run_modelhub(self):
        if False:
            for i in range(10):
                print('nop')
        recognition_pipeline = pipeline(self.task, self.model_id)
        file_name = 'data/test/images/image_structured_model_probing_test_image.jpg'
        result = recognition_pipeline(file_name)
        print(f'recognition output: {result}.')
if __name__ == '__main__':
    unittest.main()