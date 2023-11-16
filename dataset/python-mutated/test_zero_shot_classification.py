import unittest
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.models.nlp import SbertForSequenceClassification
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp import ZeroShotClassificationPipeline
from modelscope.preprocessors import ZeroShotClassificationTransformersPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.regress_test_utils import IgnoreKeyFn, MsRegressTool
from modelscope.utils.test_utils import test_level

class ZeroShotClassificationTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.task = Tasks.zero_shot_classification
        self.model_id = 'damo/nlp_structbert_zero-shot-classification_chinese-base'
    sentence = '全新突破 解放军运20版空中加油机曝光'
    labels = ['文化', '体育', '娱乐', '财经', '家居', '汽车', '教育', '科技', '军事']
    labels_str = '文化, 体育, 娱乐, 财经, 家居, 汽车, 教育, 科技, 军事'
    template = '这篇文章的标题是{}'
    regress_tool = MsRegressTool(baseline=False)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_direct_file_download(self):
        if False:
            return 10
        cache_path = snapshot_download(self.model_id)
        tokenizer = ZeroShotClassificationTransformersPreprocessor(cache_path)
        model = SbertForSequenceClassification.from_pretrained(cache_path)
        pipeline1 = ZeroShotClassificationPipeline(model, preprocessor=tokenizer)
        pipeline2 = pipeline(Tasks.zero_shot_classification, model=model, preprocessor=tokenizer)
        print(f'sentence: {self.sentence}\npipeline1:{pipeline1(input=self.sentence, candidate_labels=self.labels)}')
        print(f'sentence: {self.sentence}\npipeline2: {pipeline2(self.sentence, candidate_labels=self.labels_str, hypothesis_template=self.template)}')
        print(f'sentence: {self.sentence}\npipeline2: {pipeline2(self.sentence, candidate_labels=self.labels, hypothesis_template=self.template)}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        if False:
            i = 10
            return i + 15
        model = Model.from_pretrained(self.model_id)
        tokenizer = ZeroShotClassificationTransformersPreprocessor(model.model_dir)
        pipeline_ins = pipeline(task=Tasks.zero_shot_classification, model=model, preprocessor=tokenizer)
        print(pipeline_ins(input=self.sentence, candidate_labels=self.labels))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        if False:
            return 10
        pipeline_ins = pipeline(task=Tasks.zero_shot_classification, model=self.model_id)
        with self.regress_tool.monitor_module_single_forward(pipeline_ins.model, 'sbert_zero_shot', compare_fn=IgnoreKeyFn('.*intermediate_act_fn')):
            print(pipeline_ins(input=self.sentence, candidate_labels=self.labels))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        if False:
            while True:
                i = 10
        pipeline_ins = pipeline(task=Tasks.zero_shot_classification)
        print(pipeline_ins(input=self.sentence, candidate_labels=self.labels))
if __name__ == '__main__':
    unittest.main()