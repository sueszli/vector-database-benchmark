import os
import shutil
import tempfile
import unittest
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer

class TestFinetuneTextGeneration(unittest.TestCase):
    test_model_id = 'PAI/nlp_gpt3_text-generation_0.35B_MoE-64'

    def setUp(self):
        if False:
            print('Hello World!')
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skip('skip since the test requires multiple GPU and takes a long time to run')
    def test_finetune_poetry(self):
        if False:
            return 10
        dataset_dict = MsDataset.load('chinese-poetry-collection')
        train_dataset = dataset_dict['train'].remap_columns({'text1': 'src_txt'})
        eval_dataset = dataset_dict['test'].remap_columns({'text1': 'src_txt'})
        max_epochs = 10
        tmp_dir = './gpt_moe_poetry'
        num_warmup_steps = 100

        def noam_lambda(current_step: int):
            if False:
                i = 10
                return i + 15
            current_step += 1
            return min(current_step ** (-0.5), current_step * num_warmup_steps ** (-1.5))

        def cfg_modify_fn(cfg):
            if False:
                i = 10
                return i + 15
            cfg.train.lr_scheduler = {'type': 'LambdaLR', 'lr_lambda': noam_lambda, 'options': {'by_epoch': False}}
            cfg.train.optimizer = {'type': 'AdamW', 'lr': 0.0003}
            cfg.train.dataloader = {'batch_size_per_gpu': 1, 'workers_per_gpu': 1}
            return cfg
        kwargs = dict(model=self.test_model_id, train_dataset=train_dataset, eval_dataset=eval_dataset, max_epochs=max_epochs, work_dir=tmp_dir, cfg_modify_fn=cfg_modify_fn)
        trainer = build_trainer(name=Trainers.gpt_moe_trainer, default_args=kwargs)
        trainer.train()

    @unittest.skip('skip since the test requires multiple GPU and takes a long time to run')
    def test_finetune_dureader(self):
        if False:
            while True:
                i = 10
        dataset_dict = MsDataset.load('DuReader_robust-QG')
        train_dataset = dataset_dict['train'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'}).map(lambda example: {'src_txt': example['src_txt'].replace('[SEP]', '<sep>') + '\n'})
        eval_dataset = dataset_dict['validation'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'}).map(lambda example: {'src_txt': example['src_txt'].replace('[SEP]', '<sep>') + '\n'})
        max_epochs = 10
        tmp_dir = './gpt_moe_dureader'
        num_warmup_steps = 200

        def noam_lambda(current_step: int):
            if False:
                print('Hello World!')
            current_step += 1
            return min(current_step ** (-0.5), current_step * num_warmup_steps ** (-1.5))

        def cfg_modify_fn(cfg):
            if False:
                for i in range(10):
                    print('nop')
            cfg.train.lr_scheduler = {'type': 'LambdaLR', 'lr_lambda': noam_lambda, 'options': {'by_epoch': False}}
            cfg.train.optimizer = {'type': 'AdamW', 'lr': 0.0003}
            cfg.train.dataloader = {'batch_size_per_gpu': 16, 'workers_per_gpu': 1}
            cfg.train.hooks.append({'type': 'EvaluationHook', 'by_epoch': True, 'interval': 1})
            cfg.preprocessor.sequence_length = 512
            cfg.model.checkpoint_model_parallel_size = 1
            return cfg
        kwargs = dict(model=self.test_model_id, train_dataset=train_dataset, eval_dataset=eval_dataset, max_epochs=max_epochs, work_dir=tmp_dir, cfg_modify_fn=cfg_modify_fn)
        trainer = build_trainer(name=Trainers.gpt_moe_trainer, default_args=kwargs)
        trainer.train()
if __name__ == '__main__':
    unittest.main()