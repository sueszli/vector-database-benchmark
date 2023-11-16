import pathlib
from typing import Any, Mapping, Union
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
(_, tf, _) = try_import_tf()

class TfRLModule(tf.keras.Model, RLModule):
    """Base class for RLlib TensorFlow RLModules."""
    framework = 'tf2'

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        tf.keras.Model.__init__(self)
        RLModule.__init__(self, *args, **kwargs)

    def call(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        if False:
            while True:
                i = 10
        'Forward pass of the module.\n\n        Note:\n            This is aliased to forward_train to follow the Keras Model API.\n\n        Args:\n            batch: The input batch. This input batch should comply with\n                input_specs_train().\n            **kwargs: Additional keyword arguments.\n\n        Returns:\n            The output of the forward pass. This output should comply with the\n            ouptut_specs_train().\n\n        '
        return self.forward_train(batch)

    @override(RLModule)
    def get_state(self) -> Mapping[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return self.get_weights()

    @override(RLModule)
    def set_state(self, state_dict: Mapping[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.set_weights(state_dict)

    @override(RLModule)
    def _module_state_file_name(self) -> pathlib.Path:
        if False:
            print('Hello World!')
        return pathlib.Path('module_state')

    @override(RLModule)
    def save_state(self, dir: Union[str, pathlib.Path]) -> None:
        if False:
            return 10
        'Saves the weights of this RLModule to the directory dir.\n\n        Args:\n            dir: The directory to save the checkpoint to.\n\n        NOTE: For this TfRLModule, we save the weights in the TF checkpoint\n            format, so the file name should have no ending and should be a plain string.\n            e.g. "my_checkpoint" instead of "my_checkpoint.h5". This method of\n            checkpointing saves the module weights as multiple files, so we recommend\n            passing a file path relative to a directory, e.g.\n            "my_checkpoint/module_state".\n\n        '
        path = str(pathlib.Path(dir) / self._module_state_file_name())
        self.save_weights(path, save_format='tf')

    @override(RLModule)
    def load_state(self, dir: Union[str, pathlib.Path]) -> None:
        if False:
            return 10
        path = str(pathlib.Path(dir) / self._module_state_file_name())
        self.load_weights(path)