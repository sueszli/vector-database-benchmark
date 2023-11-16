import os
import shutil
import tempfile
import unittest
import torch
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.trainers import build_trainer
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import DistributedTestCase, test_level

@unittest.skipIf(not torch.cuda.is_available() or torch.cuda.device_count() <= 1, 'distributed unittest')
class TestGPT3OneLayerBaseQAandCWTP(DistributedTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        print('Testing %s.%s' % (type(self).__name__, self._testMethodName))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        if False:
            print('Hello World!')
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_finetune_poetry_tp_1(self):
        if False:
            return 10
        self.start(finetune_poetry_tp_1, num_gpus=gpt3_one_layer_cw_tp_1.tp, dist_start_cmd=gpt3_one_layer_cw_tp_1.dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_1_layer_evaluate_poetry_tp_1(self):
        if False:
            print('Hello World!')
        self.start(evaluate_poetry_tp_1, num_gpus=gpt3_one_layer_cw_tp_1.tp, dist_start_cmd=gpt3_one_layer_cw_tp_1.dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_1_layer_predict_poetry_tp_1(self):
        if False:
            i = 10
            return i + 15
        self.start(predict_poetry_tp_1, num_gpus=gpt3_one_layer_cw_tp_1.tp, dist_start_cmd=gpt3_one_layer_cw_tp_1.dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_1_layer_output_pipeline_poetry_tp_1(self):
        if False:
            print('Hello World!')
        self.start(pipeline_poetry_tp_1, num_gpus=gpt3_one_layer_cw_tp_1.tp, dist_start_cmd=gpt3_one_layer_cw_tp_1.dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_finetune_dureader_tp_1(self):
        if False:
            for i in range(10):
                print('nop')
        self.start(finetune_dureader_tp_1, num_gpus=gpt3_one_layer_qa_tp_1.tp, dist_start_cmd=gpt3_one_layer_qa_tp_1.dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_1_layer_evaluate_dureader_tp_1(self):
        if False:
            i = 10
            return i + 15
        self.start(evaluate_dureader_tp_1, num_gpus=gpt3_one_layer_qa_tp_1.tp, dist_start_cmd=gpt3_one_layer_qa_tp_1.dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_1_layer_predict_dureader_tp_1(self):
        if False:
            i = 10
            return i + 15
        self.start(predict_dureader_tp_1, num_gpus=gpt3_one_layer_qa_tp_1.tp, dist_start_cmd=gpt3_one_layer_qa_tp_1.dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_1_layer_output_pipeline_dureader_tp_1(self):
        if False:
            return 10
        self.start(pipeline_dureader_tp_1, num_gpus=gpt3_one_layer_qa_tp_1.tp, dist_start_cmd=gpt3_one_layer_qa_tp_1.dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_finetune_poetry_tp_2(self):
        if False:
            while True:
                i = 10
        self.start(finetune_poetry_tp_2, num_gpus=gpt3_one_layer_cw_tp_2.tp, dist_start_cmd=gpt3_one_layer_cw_tp_2.dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_1_layer_evaluate_poetry_tp_2(self):
        if False:
            for i in range(10):
                print('nop')
        self.start(evaluate_poetry_tp_2, num_gpus=gpt3_one_layer_cw_tp_2.tp, dist_start_cmd=gpt3_one_layer_cw_tp_2.dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_1_layer_predict_poetry_tp_2(self):
        if False:
            return 10
        self.start(predict_poetry_tp_2, num_gpus=gpt3_one_layer_cw_tp_2.tp, dist_start_cmd=gpt3_one_layer_cw_tp_2.dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_1_layer_output_pipeline_poetry_tp_2(self):
        if False:
            while True:
                i = 10
        self.start(pipeline_poetry_tp_2, num_gpus=gpt3_one_layer_cw_tp_2.tp, dist_start_cmd=gpt3_one_layer_cw_tp_2.dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_finetune_dureader_tp_2(self):
        if False:
            i = 10
            return i + 15
        self.start(finetune_dureader_tp_2, num_gpus=gpt3_one_layer_qa_tp_2.tp, dist_start_cmd=gpt3_one_layer_qa_tp_2.dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_1_layer_evaluate_dureader_tp_2(self):
        if False:
            while True:
                i = 10
        self.start(evaluate_dureader_tp_2, num_gpus=gpt3_one_layer_qa_tp_2.tp, dist_start_cmd=gpt3_one_layer_qa_tp_2.dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_1_layer_predict_dureader_tp_2(self):
        if False:
            i = 10
            return i + 15
        self.start(predict_dureader_tp_2, num_gpus=gpt3_one_layer_qa_tp_2.tp, dist_start_cmd=gpt3_one_layer_qa_tp_2.dist_start_cmd)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_1_layer_output_pipeline_dureader_tp_2(self):
        if False:
            print('Hello World!')
        self.start(pipeline_dureader_tp_2, num_gpus=gpt3_one_layer_qa_tp_2.tp, dist_start_cmd=gpt3_one_layer_qa_tp_2.dist_start_cmd)

class GPT3OneLayerQAandCW:

    def __init__(self, dataset_name, tp, work_dir):
        if False:
            return 10
        self.dataset_name = dataset_name
        self.tp = tp
        self.dist_start_cmd = f'torchrun --nproc_per_node {self.tp}'
        dataset_dict = MsDataset.load(dataset_name)
        if dataset_name == 'DuReader_robust-QG':
            self.train_dataset = dataset_dict['train'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'}).map(lambda example: {'src_txt': example['src_txt'].replace('[SEP]', '<sep>') + '\n'}).select(range(20))
            self.eval_dataset = dataset_dict['validation'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'}).map(lambda example: {'src_txt': example['src_txt'].replace('[SEP]', '<sep>') + '\n'}).select(range(20))
        if dataset_name == 'chinese-poetry-collection':
            self.train_dataset = dataset_dict['train'].remap_columns({'text1': 'src_txt'}).select(range(20))
            self.eval_dataset = dataset_dict['test'].remap_columns({'text1': 'src_txt'}).select(range(20))
        self.tp = tp
        self.work_dir = work_dir

    def finetune(self, max_epochs, cfg_modify_fn):
        if False:
            print('Hello World!')
        kwargs = dict(model='damo/nlp_gpt3_text-generation_1.3B', train_dataset=self.train_dataset, eval_dataset=self.eval_dataset, max_epochs=max_epochs, work_dir=self.work_dir, cfg_modify_fn=cfg_modify_fn)
        trainer = build_trainer(name=Trainers.gpt3_trainer, default_args=kwargs)
        trainer.train()

    def evaluate(self):
        if False:
            i = 10
            return i + 15
        kwargs = dict(model=f'{self.work_dir}/output', eval_dataset=self.eval_dataset, work_dir=self.work_dir)
        trainer = build_trainer(default_args=kwargs)
        trainer.evaluate()

    def predict(self):
        if False:
            print('Hello World!')
        kwargs = dict(model=f'{self.work_dir}/output', predict_datasets=self.eval_dataset, work_dir=self.work_dir)
        trainer = build_trainer(default_args=kwargs)
        trainer.predict()

    def pipeline(self):
        if False:
            return 10
        pipeline_ins = pipeline(Tasks.text_generation, model=f'{self.work_dir}/output', work_dir=self.work_dir)
        return pipeline_ins
gpt3_one_layer_cw_tp_1 = GPT3OneLayerQAandCW(dataset_name='chinese-poetry-collection', tp=1, work_dir='./gpt3_poetry_tp_1')

def finetune_poetry_helper(max_epochs, gpt3_one_layer_qa_cw_obj):
    if False:
        i = 10
        return i + 15
    max_epochs = max_epochs
    num_warmup_steps = 100

    def noam_lambda(current_step: int):
        if False:
            while True:
                i = 10
        current_step += 1
        return min(current_step ** (-0.5), current_step * num_warmup_steps ** (-1.5))

    def cfg_modify_fn(cfg):
        if False:
            return 10
        cfg.train.lr_scheduler = {'type': 'LambdaLR', 'lr_lambda': noam_lambda, 'options': {'by_epoch': False}}
        cfg.train.optimizer = {'type': 'AdamW', 'lr': 0.0003}
        cfg.train.dataloader = {'batch_size_per_gpu': 2, 'workers_per_gpu': 1}
        cfg.train.hooks.append({'type': 'MegatronHook'})
        cfg.evaluation.dataloader = {'batch_size_per_gpu': 2, 'workers_per_gpu': 1}
        cfg.evaluation.metrics = 'ppl'
        cfg.num_hidden_layers = 1
        cfg.megatron.world_size = gpt3_one_layer_qa_cw_obj.tp
        cfg.megatron.tensor_model_parallel_size = gpt3_one_layer_qa_cw_obj.tp
        return cfg
    gpt3_one_layer_qa_cw_obj.finetune(max_epochs=max_epochs, cfg_modify_fn=cfg_modify_fn)

def finetune_poetry_tp_1():
    if False:
        while True:
            i = 10
    finetune_poetry_helper(max_epochs=2, gpt3_one_layer_qa_cw_obj=gpt3_one_layer_cw_tp_1)

def evaluate_poetry_tp_1():
    if False:
        for i in range(10):
            print('nop')
    gpt3_one_layer_cw_tp_1.evaluate()

def predict_poetry_tp_1():
    if False:
        while True:
            i = 10
    gpt3_one_layer_cw_tp_1.predict()

def pipeline_poetry_helper(gpt3_one_layer_qa_cw_obj):
    if False:
        while True:
            i = 10
    pipe = gpt3_one_layer_qa_cw_obj.pipeline()
    input = '窗含西岭千秋雪'
    gen_content = pipe(input, max_length=128)
    with open(f'{gpt3_one_layer_qa_cw_obj.work_dir}/\\\n            gpt3_1_layer_base_tp_{gpt3_one_layer_qa_cw_obj.tp}_cw_pipeline_gen_text.txt', 'w', encoding='utf-8') as f:
        f.write(gen_content)

def pipeline_poetry_tp_1():
    if False:
        print('Hello World!')
    pipeline_poetry_helper(gpt3_one_layer_qa_cw_obj=gpt3_one_layer_cw_tp_1)
gpt3_one_layer_cw_tp_2 = GPT3OneLayerQAandCW(dataset_name='chinese-poetry-collection', tp=2, work_dir='./gpt3_poetry_tp_2')

def finetune_poetry_tp_2():
    if False:
        print('Hello World!')
    finetune_poetry_helper(max_epochs=2, gpt3_one_layer_qa_cw_obj=gpt3_one_layer_cw_tp_2)

def evaluate_poetry_tp_2():
    if False:
        while True:
            i = 10
    gpt3_one_layer_cw_tp_2.evaluate()

def predict_poetry_tp_2():
    if False:
        for i in range(10):
            print('nop')
    gpt3_one_layer_cw_tp_2.predict()

def pipeline_poetry_tp_2():
    if False:
        print('Hello World!')
    pipeline_poetry_helper(gpt3_one_layer_qa_cw_obj=gpt3_one_layer_cw_tp_2)
gpt3_one_layer_qa_tp_1 = GPT3OneLayerQAandCW(dataset_name='DuReader_robust-QG', tp=1, work_dir='./dureader_tp_1')

def finetune_dureader_helper(max_epochs, gpt3_one_layer_qa_cw_obj):
    if False:
        return 10
    max_epochs = max_epochs
    num_warmup_steps = 100

    def noam_lambda(current_step: int):
        if False:
            print('Hello World!')
        current_step += 1
        return min(current_step ** (-0.5), current_step * num_warmup_steps ** (-1.5))

    def cfg_modify_fn(cfg):
        if False:
            i = 10
            return i + 15
        cfg.train.lr_scheduler = {'type': 'LambdaLR', 'lr_lambda': noam_lambda, 'options': {'by_epoch': False}}
        cfg.train.optimizer = {'type': 'AdamW', 'lr': 0.0003}
        cfg.train.dataloader = {'batch_size_per_gpu': 16, 'workers_per_gpu': 1}
        cfg.train.hooks.append({'type': 'EvaluationHook', 'by_epoch': True, 'interval': 1})
        cfg.train.hooks.append({'type': 'MegatronHook'})
        cfg.num_hidden_layers = 1
        cfg.preprocessor.sequence_length = 512
        cfg.model.checkpoint_model_parallel_size = 1
        cfg.megatron.world_size = gpt3_one_layer_qa_cw_obj.tp
        cfg.megatron.tensor_model_parallel_size = gpt3_one_layer_qa_cw_obj.tp
        return cfg
    gpt3_one_layer_qa_cw_obj.finetune(max_epochs=max_epochs, cfg_modify_fn=cfg_modify_fn)

def finetune_dureader_tp_1():
    if False:
        while True:
            i = 10
    finetune_dureader_helper(max_epochs=2, gpt3_one_layer_qa_cw_obj=gpt3_one_layer_qa_tp_1)

def evaluate_dureader_tp_1():
    if False:
        print('Hello World!')
    gpt3_one_layer_qa_tp_1.evaluate()

def predict_dureader_tp_1():
    if False:
        return 10
    gpt3_one_layer_qa_tp_1.predict()

def pipeline_dureader_helper(gpt3_one_layer_qa_cw_obj):
    if False:
        for i in range(10):
            print('nop')
    pipe = gpt3_one_layer_qa_cw_obj.pipeline()
    input1 = '推荐一下防脱发的洗发水'
    input2 = '重新推荐一次'
    gen_content1 = pipe(input1, max_length=128)
    with open(f'{gpt3_one_layer_qa_cw_obj.work_dir}/\\\n            gpt3_1_layer_base_tp_{gpt3_one_layer_qa_cw_obj.tp}_qa_pipeline_gen_text.txt', 'a+', encoding='utf-8') as f:
        f.write(gen_content1 + '\n')
    gen_content2 = pipe(input2, max_length=128)
    with open(f'{gpt3_one_layer_qa_cw_obj.work_dir}/\\\n            gpt3_1_layer_base_tp_{gpt3_one_layer_qa_cw_obj.tp}_qa_pipeline_gen_text.txt', 'a+', encoding='utf-8') as f:
        f.write(gen_content2 + '\n')

def pipeline_dureader_tp_1():
    if False:
        i = 10
        return i + 15
    pipeline_dureader_helper(gpt3_one_layer_qa_tp_1)
gpt3_one_layer_qa_tp_2 = GPT3OneLayerQAandCW(dataset_name='DuReader_robust-QG', tp=2, work_dir='./dureader_tp_2')

def finetune_dureader_tp_2():
    if False:
        print('Hello World!')
    finetune_dureader_helper(max_epochs=2, gpt3_one_layer_qa_cw_obj=gpt3_one_layer_qa_tp_2)

def evaluate_dureader_tp_2():
    if False:
        print('Hello World!')
    gpt3_one_layer_qa_tp_2.evaluate()

def predict_dureader_tp_2():
    if False:
        i = 10
        return i + 15
    gpt3_one_layer_qa_tp_2.predict()

def pipeline_dureader_tp_2():
    if False:
        for i in range(10):
            print('nop')
    pipeline_dureader_helper(gpt3_one_layer_qa_tp_2)

class CustomTestLoader(unittest.TestLoader):

    def getTestCaseNames(self, testcase_class):
        if False:
            while True:
                i = 10
        test_names = super().getTestCaseNames(testcase_class)
        testcase_methods = list(testcase_class.__dict__.keys())
        test_names.sort(key=testcase_methods.index)
        return test_names
if __name__ == '__main__':
    unittest.main(testLoader=CustomTestLoader())