import os
import time
from typing import Callable, Dict, Optional, Tuple, Union
import numpy as np
from modelscope.metainfo import Trainers
from modelscope.models.nlp.space.model.generator import SpaceGenerator
from modelscope.models.nlp.space.model.model_base import SpaceModelBase
from modelscope.preprocessors.nlp import MultiWOZBPETextField
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.nlp.space.eval import MultiWOZEvaluator
from modelscope.trainers.nlp.space.trainer.gen_trainer import MultiWOZTrainer
from modelscope.utils.config import Config, ModelFile
from modelscope.utils.logger import get_logger
logger = get_logger()

def setup_seed(seed: int):
    if False:
        return 10
    import random
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

@TRAINERS.register_module(module_name=Trainers.dialog_modeling_trainer)
class DialogModelingTrainer(BaseTrainer):

    def __init__(self, cfg_file: Optional[str]=None, cfg_modify_fn: Optional[Callable]=None, *args, **kwargs):
        if False:
            return 10
        super().__init__(os.path.join(kwargs['model_dir'], kwargs['cfg_name']))
        self.cfg_modify_fn = cfg_modify_fn
        self.cfg = self.rebuild_config(self.cfg)
        setup_seed(self.cfg.Trainer.seed)
        self.bpe = MultiWOZBPETextField(self.cfg, **kwargs)
        self.cfg.Model.num_token_embeddings = self.bpe.vocab_size
        self.cfg.Model.num_turn_embeddings = self.bpe.max_ctx_turn + 1
        if 'work_dir' in kwargs:
            self.cfg.Trainer.save_dir = kwargs['work_dir']
        else:
            self.cfg.Trainer.save_dir = './default_save_dir'
        self.train_data = self.bpe.get_batches('train')
        self.dev_data = self.bpe.get_batches('dev')
        self.evaluator = MultiWOZEvaluator(reader=self.bpe, **kwargs)
        self.generator = SpaceGenerator.create(self.cfg, reader=self.bpe)
        self._load_model(**kwargs)

    def _load_model(self, **kwargs):
        if False:
            i = 10
            return i + 15

        def to_tensor(array):
            if False:
                print('Hello World!')
            '\n            numpy array -> tensor\n            '
            import torch
            array = torch.tensor(array)
            return array.cuda() if self.cfg.use_gpu and torch.cuda.is_available() else array
        if 'model' in kwargs:
            self.model = kwargs['model']
        else:
            self.model = SpaceModelBase.create(kwargs['model_dir'], self.cfg, reader=self.bpe, generator=self.generator)
        import torch
        if self.cfg.Trainer.gpu > 1 and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.trainer = MultiWOZTrainer(self.model, to_tensor, self.cfg, reader=self.bpe, evaluator=self.evaluator)
        self.trainer.set_optimizers()
        self.trainer.load()

    def rebuild_config(self, cfg: Config):
        if False:
            print('Hello World!')
        if self.cfg_modify_fn is not None:
            return self.cfg_modify_fn(cfg)
        return cfg

    def train(self, *args, **kwargs):
        if False:
            return 10
        logger.info('Train')
        self.trainer.train(train_data=self.train_data, dev_data=self.dev_data)

    def evaluate(self, checkpoint_path: Optional[str]=None, *args, **kwargs) -> Dict[str, float]:
        if False:
            i = 10
            return i + 15
        logger.info('Evaluate')
        self.cfg.do_infer = True
        pos = checkpoint_path.rfind('/')
        checkpoint_name = checkpoint_path[pos + 1:]
        checkpoint_dir = checkpoint_path[:pos]
        assert checkpoint_name == ModelFile.TORCH_MODEL_BIN_FILE
        kwargs['model_dir'] = checkpoint_dir
        self._load_model(**kwargs)
        self.trainer.infer(data_type='test')