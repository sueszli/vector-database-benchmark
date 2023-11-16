import os
import shutil
import tempfile
import unittest
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import DownloadMode
from modelscope.utils.test_utils import test_level

class TestStableDiffusionTrainer(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        print('Testing %s.%s' % (type(self).__name__, self._testMethodName))
        self.train_dataset = MsDataset.load('buptwq/lora-stable-diffusion-finetune', split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD)
        self.eval_dataset = MsDataset.load('buptwq/lora-stable-diffusion-finetune', split='validation', download_mode=DownloadMode.FORCE_REDOWNLOAD)
        self.max_epochs = 5
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        if False:
            print('Hello World!')
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_stable_diffusion_train(self):
        if False:
            print('Hello World!')
        model_id = 'AI-ModelScope/stable-diffusion-v1-5'
        model_revision = 'v1.0.7'

        def cfg_modify_fn(cfg):
            if False:
                return 10
            cfg.train.max_epochs = self.max_epochs
            cfg.train.lr_scheduler = {'type': 'LambdaLR', 'lr_lambda': lambda _: 1, 'last_epoch': -1}
            cfg.train.optimizer.lr = 0.0001
            return cfg
        kwargs = dict(model=model_id, model_revision=model_revision, work_dir=self.tmp_dir, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset, cfg_modify_fn=cfg_modify_fn)
        trainer = build_trainer(name=Trainers.stable_diffusion, default_args=kwargs)
        trainer.train()
        result = trainer.evaluate()
        print(f'Stable-diffusion train output: {result}.')
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_stable_diffusion_eval(self):
        if False:
            i = 10
            return i + 15
        model_id = 'AI-ModelScope/stable-diffusion-v1-5'
        model_revision = 'v1.0.7'
        kwargs = dict(model=model_id, model_revision=model_revision, work_dir=self.tmp_dir, train_dataset=None, eval_dataset=self.eval_dataset)
        trainer = build_trainer(name=Trainers.stable_diffusion, default_args=kwargs)
        result = trainer.evaluate()
        print(f'Stable-diffusion eval output: {result}.')
if __name__ == '__main__':
    unittest.main()