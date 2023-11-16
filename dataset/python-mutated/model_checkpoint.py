from bigdl.orca.learn.pytorch.callbacks import Callback
import os
import re
import warnings
from bigdl.dllib.utils.log4Error import invalidInputError

class ModelCheckpoint(Callback):
    CHECKPOINT_JOIN_CHAR = '-'
    CHECKPOINT_NAME_LAST = 'last'
    FILE_EXTENSION = '.ckpt'

    def __init__(self, filepath: str='', save_weights_only: bool=False, by_epoch: bool=True, interval: int=-1):
        if False:
            print('Hello World!')
        '\n        ModelCheckpoint callback is used in conjunction with training using estimator.fit() to save\n        a model or weights (in a checkpoint file) at some interval, so the model or weights can be\n        loaded later to continue the training from the state saved.\n        Example:\n            >>> checkpoint_callback = ModelCheckpoint(\n            ...     filepath=\'my/path/sample-mnist-{epoch:02d}\',\n            ... )\n        And checkpoints will be saved as file with path like \'my/path/sample-mnist-epoch=1.ckpt\'\n        with different epoch values.\n        :param filepath: path to save the model file.\n        :param by_epoch: save chekpoint by epoch or by iteration\n        :param interval: The saving period. If ``by_epoch=True``, interval\n            indicates epochs, otherwise it indicates iterations. Default: -1, which means "never".\n        '
        super().__init__()
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.by_epoch = by_epoch
        self.interval = interval
        self.last_ckpt_path = ''
        self.filename = os.path.basename(self.filepath)
        self.dirname = os.path.dirname(self.filepath)

    def after_train_epoch(self, runner):
        if False:
            return 10
        '\n        Called at the end of an epoch.\n        Subclasses should override for any actions to run. This function should only\n        be called during TRAIN mode.\n        :param epoch:  Integer, index of epoch.\n        '
        if not self.by_epoch:
            return
        if self.interval < 0:
            self.interval = 1
        if self.every_n_epoch(runner, self.interval):
            stats = {'epoch': runner.epochs}
            last_ckpt_path = self._format_checkpoint_name(dirname=self.dirname, filename=self.filename, stats=stats)
            runner.save_checkpoint(last_ckpt_path, self.save_weights_only)

    def after_train_iter(self, runner):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called at the end of an iteration.\n        Subclasses should override for any actions to run. This function should only\n        be called during TRAIN mode.\n        '
        if self.by_epoch:
            return
        if self.every_n_iter(runner, self.interval):
            stats = {'iter': runner.global_step + 1}
            last_ckpt_path = self._format_checkpoint_name(dirname=self.dirname, filename=self.filename, stats=stats)
            runner.save_checkpoint(last_ckpt_path, self.save_weights_only)

    def before_run(self, runner):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called at the beginning of training.\n        Subclasses should override for any actions to run.\n        :param logs: Dict. Currently, no data is passed to this argument for this method\n          but that may change in the future.\n        '
        dirname = os.path.dirname(self.filepath)
        from bigdl.orca.data.file import exists, listdir, makedirs
        if exists(dirname):
            files = [os.path.basename(f) for f in listdir(dirname)]
            files = [x for x in files if 'ckpt' in x]
            if len(files) == 0:
                return None
            invalidInputError(False, f'Find non-empty dirname with filepath of {self.filepath}.')
        else:
            makedirs(dirname)

    def after_run(self, runner):
        if False:
            while True:
                i = 10
        '\n        Called at the end of training.\n        Subclasses should override for any actions to run.\n        '
        last_ckpt_path = self._format_checkpoint_name(dirname=self.dirname, filename=self.CHECKPOINT_NAME_LAST)
        (previous, self.last_ckpt_path) = (self.last_ckpt_path, last_ckpt_path)
        runner.save_checkpoint(last_ckpt_path, self.save_weights_only)
        if previous and previous != last_ckpt_path:
            runner.remove_checkpoint(previous)

    @classmethod
    def get_latest_checkpoint(cls, dirname):
        if False:
            return 10
        '\n        Finds the filepath of latest saved checkpoint file.\n        :param dirname: directory where the checkpoints were saved\n        return: The full path to the latest checkpoint or `None` if no checkpoint was found.\n        '
        ckpt_path = cls._format_checkpoint_name(dirname, filename=cls.CHECKPOINT_NAME_LAST)
        from bigdl.orca.data.file import exists
        if not exists(ckpt_path):
            invalidInputError(False, f'Latest checkpoint at {ckpt_path} not found.')
        return ckpt_path

    @classmethod
    def _format_checkpoint_name(cls, dirname, filename, stats=None):
        if False:
            return 10
        "\n        checkpoint name is in the format of 'epoch={epoch}.ckpt'\n        "
        groups = re.findall('(\\{.*?)[:\\}]', filename)
        if len(groups) >= 0:
            stats = dict() if stats is None else stats
            for group in groups:
                name = group[1:]
                if 'epoch' not in name and 'iter' not in name:
                    warnings.warn('We only support filepath with {epoch} or {iter} for now.')
                filename = filename.replace(group, name + '={' + name)
                if name not in stats:
                    stats[name] = 0
            filename = filename.format(**stats)
        ckpt_name = f'{filename}{cls.FILE_EXTENSION}'
        ckpt_path = os.path.join(dirname, ckpt_name)
        return ckpt_path