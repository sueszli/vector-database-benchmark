import unittest
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import SbertForFaqRanking, SbertForFaqRetrieval
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import FaqPipeline
from modelscope.preprocessors import FaqPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

class FaqTest(unittest.TestCase):
    model_id = '/Users/tanfan/Desktop/Workdir/Gitlab/maas/MaaS-lib/.faq_test_model'
    param = {'query_set': ['明天星期几', '今天星期六', '今天星期六'], 'support_set': [{'text': '今天星期六', 'label': 'label0'}, {'text': '明天星期几', 'label': 'label1'}]}

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        if False:
            for i in range(10):
                print('nop')
        model = Model.from_pretrained(self.model_id)
        preprocessor = FaqPreprocessor(model.model_dir)
        pipeline_ins = pipeline(task=Tasks.faq, model=model, preprocessor=preprocessor)
        result = pipeline_ins(self.param)
        print(result)
if __name__ == '__main__':
    unittest.main()