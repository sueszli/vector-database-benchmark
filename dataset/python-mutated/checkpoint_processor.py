import os
import re
import shutil
from modelscope.metainfo import Pipelines
from modelscope.utils.checkpoint import load_checkpoint, save_checkpoint, save_configuration
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import is_master

class CheckpointProcessor:
    TRAINER_STATE_SUFFIX = '_trainer_state.pth'
    MODEL_STATE_SUFFIX = '.pth'

    def prepare_output(self, trainer, output_dir):
        if False:
            while True:
                i = 10
        "Prepares the output of target folder.\n\n        This is a strategic function which can be registered by other hook's function.\n\n        Args:\n            trainer: The trainer instance.\n            output_dir: The target folder used in inference.\n        "
        config = trainer.cfg
        if config['task'] in [getattr(Pipelines, attr) for attr in dir(Pipelines) if not attr.startswith('__')]:
            config['pipeline'] = {'type': config['task']}
        self.copy_files_and_dump_config(trainer, output_dir, config, '*.bin')

    @staticmethod
    def copy_files_and_dump_config(trainer, output_dir, config, bin_file):
        if False:
            i = 10
            return i + 15
        'Copy useful files to target output folder and dumps the target configuration.json.\n        '
        model = trainer.unwrap_module(trainer.model)

        class SaveConfig:

            def __init__(self, output_dir, config):
                if False:
                    return 10
                self.output_dir = output_dir
                self.config = config

            def __call__(self, _output_dir, _config):
                if False:
                    while True:
                        i = 10
                self.config = _config

            def save_config(self):
                if False:
                    for i in range(10):
                        print('nop')
                save_configuration(self.output_dir, self.config)
        for pop_key in ['push_to_hub', 'hub_repo_id', 'hub_token', 'private_hub']:
            if config.safe_get('train.checkpoint.period.' + pop_key) is not None:
                config.safe_get('train.checkpoint.period').pop(pop_key)
            if config.safe_get('train.checkpoint.best.' + pop_key) is not None:
                config.safe_get('train.checkpoint.best').pop(pop_key)
        save_config_fn = SaveConfig(output_dir, config)
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(output_dir, bin_file, save_function=lambda *args, **kwargs: None, config=save_config_fn.config, save_config_function=save_config_fn)
        if trainer.train_preprocessor is not None:
            trainer.train_preprocessor.save_pretrained(output_dir, save_config_fn.config, save_config_function=save_config_fn)
        if trainer.eval_preprocessor is not None:
            trainer.eval_preprocessor.save_pretrained(output_dir, save_config_fn.config, save_config_function=save_config_fn)
        save_config_fn.save_config()

    @staticmethod
    def _bin_file(model):
        if False:
            while True:
                i = 10
        'Get bin file path.\n        '
        default_bin_file = ModelFile.TORCH_MODEL_BIN_FILE
        if hasattr(model, 'model_dir') and ModelFile.TORCH_MODEL_FILE in os.listdir(model.model_dir):
            default_bin_file = ModelFile.TORCH_MODEL_FILE
        return default_bin_file

    def save_checkpoints(self, trainer, checkpoint_path_prefix, output_dir, meta=None, save_optimizers=True):
        if False:
            i = 10
            return i + 15
        "Save the state dict for trainer and model.\n\n        This is a strategic function which can be registered by other hook's function.\n\n        Args:\n            trainer(`EpochBasedTrainer`): The trainer instance.\n            checkpoint_path_prefix(`str`): The saving dir with a prefix.\n                like: /tmp/test/epoch_0\n            output_dir(`str`): The output dir for inference.\n            meta: (`dict`): The meta info needed to be saved into files.\n            save_optimizers: (`bool`): Do save the optimizers state\n        "
        model = trainer.unwrap_module(trainer.model)
        (_model_file, _train_state_file) = self._get_state_file_name(checkpoint_path_prefix)
        self.save_trainer_state(trainer, model, _train_state_file, meta, save_optimizers)
        self.save_model_state(model, _model_file)
        self.link(model, _model_file, output_dir)

    def remove_checkpoints(self, trainer, checkpoint_path_prefix):
        if False:
            while True:
                i = 10
        "Remove obsolete checkpoint files.\n\n        This is a strategic function which can be registered by other hook's function.\n\n        Args:\n            trainer(`EpochBasedTrainer`): The trainer instance.\n            checkpoint_path_prefix(`str`): The saving dir with a prefix.\n                like: /tmp/test/epoch_0\n        "
        (_model_file, _train_state_file) = self._get_state_file_name(checkpoint_path_prefix)
        if os.path.isfile(_train_state_file):
            os.remove(_train_state_file)
        if os.path.isfile(_model_file):
            os.remove(_model_file)

    def should_save_on_rank(self, trainer):
        if False:
            while True:
                i = 10
        "Used in ddp or other distributed training scenario, returns whether do saving in current rank.\n\n        This is a strategic function which can be registered by other hook's function.\n\n        Args:\n            trainer(`EpochBasedTrainer`): The trainer instance.\n        "
        return is_master()

    def link(self, model, src_file, output_dir):
        if False:
            return 10
        'Links the src bin file to the output folder.\n\n        Args:\n            model: The model instance.\n            src_file: The src bin file path.\n            output_dir: The target folder used in inference.\n        '
        bin_file = self._bin_file(model)
        dest_file = os.path.join(output_dir, bin_file)
        if os.path.isfile(dest_file):
            os.unlink(dest_file)
        try:
            os.link(src_file, dest_file)
        except OSError as e:
            get_logger().error(f'Link {src_file} to {dest_file} error: {e}, changing to copy the bin file, this may use more disk space.')
            shutil.copyfile(src_file, dest_file)

    def save_trainer_state(self, trainer, model, train_state_file, meta, save_optimizers):
        if False:
            for i in range(10):
                print('nop')
        "Save the trainer state, including optimizer/lr_scheduler's state dict, random states etc.\n\n        Args:\n            trainer: The trainer instance.\n            model: The model instance.\n            train_state_file: The target file name for saving trainer states.\n            meta: Some extra meta info.\n            save_optimizers: Save optimizers state or not.\n        "
        save_checkpoint(model, train_state_file, trainer.optimizer if save_optimizers else None, trainer.lr_scheduler if save_optimizers else None, meta=meta, with_model=False)

    def save_model_state(self, model, model_file):
        if False:
            i = 10
            return i + 15
        'Save the model state.\n\n        Args:\n            model: The model instance.\n            model_file: The target file name for saving model states.\n        '
        save_checkpoint(model, model_file, None, None, meta=None, with_meta=False)

    def load_checkpoints(self, checkpoint_path_prefix, trainer, load_all_state, strict):
        if False:
            return 10
        "Load checkpoint files of trainer state and model state.\n\n        This is a strategic function which can be registered by other hook's function.\n\n        Args:\n            checkpoint_path_prefix(str): The checkpoint dir with prefix or a model state file.\n                Example: '/tmp/test/epoch_0' or '/tmp/test/epoch_0.pth'\n            trainer(`EpochBasedTrainer`): The trainer instance.\n            load_all_state(`boolean`): Load all states (else load only module states).\n            strict(`boolean`): If strict, any unmatched keys will cause an error.\n\n        Returns:\n            The meta info in json.\n        "
        (_model_file, _train_state_file) = self._get_state_file_name(checkpoint_path_prefix)
        meta = {}
        if os.path.isfile(_train_state_file):
            meta = self.load_trainer_state(trainer, _train_state_file, load_all_state)
        else:
            print(f'No trainer state file {_train_state_file} found, skip.')
        self.load_model_state(trainer, _model_file, strict)
        return meta

    @staticmethod
    def load_trainer_state(trainer, train_state_file, load_all_state):
        if False:
            while True:
                i = 10
        'Load trainer state file.\n        '
        optimizer = getattr(trainer, 'optimizer', None) if load_all_state else None
        lr_scheduler = getattr(trainer, 'lr_scheduler', None) if load_all_state else None
        return load_checkpoint(train_state_file, None, optimizer, lr_scheduler)

    def load_model_state(self, trainer, model_file, strict):
        if False:
            i = 10
            return i + 15
        'Load model state file.\n        '
        return load_checkpoint(model_file, trainer.unwrap_module(trainer.model), None, None)

    @staticmethod
    def _get_state_file_name(checkpoint_path_prefix):
        if False:
            for i in range(10):
                print('nop')
        "Get the default file name for state files.\n\n        If the input is a checkpoint dir with prefix, this function will append suffix for both checkpoint files.\n        If the input is an absolute file name, this function will return it as the model file name, and append\n            suffix for the trainer file name.\n\n        NOTE: a best checkpoint filename with float or int metric value inside\n            will not be judged as having a extension file name. like: '/tmp/test/epoch_0_accuracy0.85'\n\n        Args:\n            checkpoint_path_prefix(`str`): The checkpoint dir with prefix or a model state file\n            with extension file name. like: '/tmp/test/epoch_0'\n\n        Returns:\n              A tuple of model state file name and trainer state file name.\n        "
        (base, ext) = os.path.splitext(checkpoint_path_prefix)
        if len(ext) == 0 or re.match('^\\d+$', ext[1:]):
            return (checkpoint_path_prefix + CheckpointProcessor.MODEL_STATE_SUFFIX, checkpoint_path_prefix + CheckpointProcessor.TRAINER_STATE_SUFFIX)
        else:
            return (checkpoint_path_prefix, base + CheckpointProcessor.TRAINER_STATE_SUFFIX.split('.')[0] + '.' + ext[1:])