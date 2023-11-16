from typing import Any
from torch.utils.tensorboard import SummaryWriter
from snorkel.types import Config
from .log_writer import LogWriter

class TensorBoardWriter(LogWriter):
    """A class for logging to Tensorboard during training process.

    See ``LogWriter`` for more attributes.

    Parameters
    ----------
    kwargs
        Passed to ``LogWriter`` initializer

    Attributes
    ----------
    writer
        ``SummaryWriter`` for logging and visualization
    """

    def __init__(self, **kwargs: Any) -> None:
        if False:
            return 10
        super().__init__(**kwargs)
        self.writer = SummaryWriter(self.log_dir)

    def add_scalar(self, name: str, value: float, step: float) -> None:
        if False:
            while True:
                i = 10
        'Log a scalar variable to TensorBoard.\n\n        Parameters\n        ----------\n        name\n            Name of the scalar collection\n        value\n            Value of scalar\n        step\n            Step axis value\n        '
        self.writer.add_scalar(name, value, step)

    def write_config(self, config: Config, config_filename: str='config.json') -> None:
        if False:
            i = 10
            return i + 15
        'Dump the config to file and add it to TensorBoard.\n\n        Parameters\n        ----------\n        config\n            JSON-compatible config to write to TensorBoard\n        config_filename\n            File to write config to\n        '
        super().write_config(config, config_filename)
        self.writer.add_text(tag='config', text_string=str(config))

    def cleanup(self) -> None:
        if False:
            while True:
                i = 10
        'Close the ``SummaryWriter``.'
        self.writer.close()