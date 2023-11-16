import os
import shutil
import tempfile
import unittest
import zipfile
from functools import partial
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import DownloadMode, ModelFile
from modelscope.utils.test_utils import test_level

class TestGeneralImageClassificationTestTrainer(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        print('Testing %s.%s' % (type(self).__name__, self._testMethodName))
        try:
            self.train_dataset = MsDataset.load('cats_and_dogs', namespace='tany0699', subset_name='default', split='train')
            self.eval_dataset = MsDataset.load('cats_and_dogs', namespace='tany0699', subset_name='default', split='validation')
        except Exception as e:
            print(f'Download dataset error: {e}')
        self.max_epochs = 1
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        if False:
            return 10
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_nextvit_dailylife_train(self):
        if False:
            return 10
        model_id = 'damo/cv_nextvit-small_image-classification_Dailylife-labels'

        def cfg_modify_fn(cfg):
            if False:
                print('Hello World!')
            cfg.train.dataloader.batch_size_per_gpu = 32
            cfg.train.dataloader.workers_per_gpu = 1
            cfg.train.max_epochs = self.max_epochs
            cfg.model.mm_model.head.num_classes = 2
            cfg.train.optimizer.lr = 0.0001
            cfg.train.lr_config.warmup_iters = 1
            cfg.train.evaluation.metric_options = {'topk': (1,)}
            cfg.evaluation.metric_options = {'topk': (1,)}
            return cfg
        kwargs = dict(model=model_id, work_dir=self.tmp_dir, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset, cfg_modify_fn=cfg_modify_fn)
        trainer = build_trainer(name=Trainers.image_classification, default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i + 1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_nextvit_dailylife_eval(self):
        if False:
            for i in range(10):
                print('nop')
        model_id = 'damo/cv_nextvit-small_image-classification_Dailylife-labels'
        kwargs = dict(model=model_id, work_dir=self.tmp_dir, train_dataset=None, eval_dataset=self.eval_dataset)
        trainer = build_trainer(name=Trainers.image_classification, default_args=kwargs)
        result = trainer.evaluate()
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_convnext_garbage_train(self):
        if False:
            i = 10
            return i + 15
        model_id = 'damo/cv_convnext-base_image-classification_garbage'

        def cfg_modify_fn(cfg):
            if False:
                return 10
            cfg.train.dataloader.batch_size_per_gpu = 16
            cfg.train.dataloader.workers_per_gpu = 1
            cfg.train.max_epochs = self.max_epochs
            cfg.model.mm_model.head.num_classes = 2
            cfg.train.optimizer.lr = 0.0001
            cfg.train.lr_config.warmup_iters = 1
            cfg.train.evaluation.metric_options = {'topk': (1,)}
            cfg.evaluation.metric_options = {'topk': (1,)}
            return cfg
        kwargs = dict(model=model_id, work_dir=self.tmp_dir, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset, cfg_modify_fn=cfg_modify_fn)
        trainer = build_trainer(name=Trainers.image_classification, default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i + 1}.pth', results_files)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_convnext_garbage_eval(self):
        if False:
            print('Hello World!')
        model_id = 'damo/cv_convnext-base_image-classification_garbage'
        kwargs = dict(model=model_id, work_dir=self.tmp_dir, train_dataset=None, eval_dataset=self.eval_dataset)
        trainer = build_trainer(name=Trainers.image_classification, default_args=kwargs)
        result = trainer.evaluate()
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_beitv2_train_eval(self):
        if False:
            print('Hello World!')
        model_id = 'damo/cv_beitv2-base_image-classification_patch16_224_pt1k_ft22k_in1k'

        def cfg_modify_fn(cfg):
            if False:
                print('Hello World!')
            cfg.train.dataloader.batch_size_per_gpu = 16
            cfg.train.dataloader.workers_per_gpu = 1
            cfg.train.max_epochs = self.max_epochs
            cfg.model.mm_model.head.num_classes = 2
            cfg.model.mm_model.head.loss.num_classes = 2
            cfg.train.optimizer.lr = 0.0001
            cfg.train.lr_config.warmup_iters = 1
            cfg.train.evaluation.metric_options = {'topk': (1,)}
            cfg.evaluation.metric_options = {'topk': (1,)}
            return cfg
        kwargs = dict(model=model_id, work_dir=self.tmp_dir, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset, cfg_modify_fn=cfg_modify_fn)
        trainer = build_trainer(name=Trainers.image_classification, default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i + 1}.pth', results_files)
        result = trainer.evaluate()
        print(result)
if __name__ == '__main__':
    unittest.main()