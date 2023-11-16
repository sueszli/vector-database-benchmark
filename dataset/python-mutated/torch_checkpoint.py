import os
import tempfile
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional
import torch
from ray.air._internal.torch_utils import consume_prefix_in_state_dict_if_present_not_in_place, load_torch_model
from ray.train._internal.framework_checkpoint import FrameworkCheckpoint
from ray.util.annotations import PublicAPI
if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor
ENCODED_DATA_KEY = 'torch_encoded_data'

@PublicAPI(stability='beta')
class TorchCheckpoint(FrameworkCheckpoint):
    """A :class:`~ray.train.Checkpoint` with Torch-specific functionality."""
    MODEL_FILENAME = 'model.pt'

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Any], *, preprocessor: Optional['Preprocessor']=None) -> 'TorchCheckpoint':
        if False:
            for i in range(10):
                print('nop')
        'Create a :class:`~ray.train.Checkpoint` that stores a model state dictionary.\n\n        .. tip::\n\n            This is the recommended method for creating\n            :class:`TorchCheckpoints<TorchCheckpoint>`.\n\n        Args:\n            state_dict: The model state dictionary to store in the checkpoint.\n            preprocessor: A fitted preprocessor to be applied before inference.\n\n        Returns:\n            A :class:`TorchCheckpoint` containing the specified state dictionary.\n\n        Examples:\n\n            .. testcode::\n\n                import torch\n                import torch.nn as nn\n                from ray.train.torch import TorchCheckpoint\n\n                # Set manual seed\n                torch.manual_seed(42)\n\n                # Function to create a NN model\n                def create_model() -> nn.Module:\n                    model = nn.Sequential(nn.Linear(1, 10),\n                            nn.ReLU(),\n                            nn.Linear(10,1))\n                    return model\n\n                # Create a TorchCheckpoint from our model\'s state_dict\n                model = create_model()\n                checkpoint = TorchCheckpoint.from_state_dict(model.state_dict())\n\n                # Now load the model from the TorchCheckpoint by providing the\n                # model architecture\n                model_from_chkpt = checkpoint.get_model(create_model())\n\n                # Assert they have the same state dict\n                assert str(model.state_dict()) == str(model_from_chkpt.state_dict())\n                print("worked")\n\n            .. testoutput::\n                :hide:\n\n                ...\n        '
        tempdir = tempfile.mkdtemp()
        model_path = os.path.join(tempdir, cls.MODEL_FILENAME)
        stripped_state_dict = consume_prefix_in_state_dict_if_present_not_in_place(state_dict, 'module.')
        torch.save(stripped_state_dict, model_path)
        checkpoint = cls.from_directory(tempdir)
        if preprocessor:
            checkpoint.set_preprocessor(preprocessor)
        return checkpoint

    @classmethod
    def from_model(cls, model: torch.nn.Module, *, preprocessor: Optional['Preprocessor']=None) -> 'TorchCheckpoint':
        if False:
            i = 10
            return i + 15
        'Create a :class:`~ray.train.Checkpoint` that stores a Torch model.\n\n        .. note::\n\n            PyTorch recommends storing state dictionaries. To create a\n            :class:`TorchCheckpoint` from a state dictionary, call\n            :meth:`~ray.train.torch.TorchCheckpoint.from_state_dict`. To learn more\n            about state dictionaries, read\n            `Saving and Loading Models <https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict>`_. # noqa: E501\n\n        Args:\n            model: The Torch model to store in the checkpoint.\n            preprocessor: A fitted preprocessor to be applied before inference.\n\n        Returns:\n            A :class:`TorchCheckpoint` containing the specified model.\n\n        Examples:\n\n            .. testcode::\n\n                from ray.train.torch import TorchCheckpoint\n                import torch\n\n                # Create model identity and send a random tensor to it\n                model = torch.nn.Identity()\n                input = torch.randn(2, 2)\n                output = model(input)\n\n                # Create a checkpoint\n                checkpoint = TorchCheckpoint.from_model(model)\n                print(checkpoint)\n\n            .. testoutput::\n                :hide:\n\n                ...\n        '
        tempdir = tempfile.mkdtemp()
        model_path = os.path.join(tempdir, cls.MODEL_FILENAME)
        torch.save(model, model_path)
        checkpoint = cls.from_directory(tempdir)
        if preprocessor:
            checkpoint.set_preprocessor(preprocessor)
        return checkpoint

    def get_model(self, model: Optional[torch.nn.Module]=None) -> torch.nn.Module:
        if False:
            for i in range(10):
                print('nop')
        'Retrieve the model stored in this checkpoint.\n\n        Args:\n            model: If the checkpoint contains a model state dict, and not\n                the model itself, then the state dict will be loaded to this\n                ``model``. Otherwise, the model will be discarded.\n        '
        with self.as_directory() as tempdir:
            model_path = os.path.join(tempdir, self.MODEL_FILENAME)
            if not os.path.exists(model_path):
                raise RuntimeError('`model.pt` not found within this checkpoint. Make sure you created this `TorchCheckpoint` from one of its public constructors (`from_state_dict` or `from_model`).')
            model_or_state_dict = torch.load(model_path, map_location='cpu')
        if isinstance(model_or_state_dict, torch.nn.Module):
            if model:
                warnings.warn('TorchCheckpoint already contains all information needed. Discarding provided `model` argument. This means: If you are using BatchPredictor, you should do `BatchPredictor.from_checkpoint(checkpoint, TorchPredictor)` byremoving kwargs `model=`. If you are using TorchPredictor directly, you should do `TorchPredictor.from_checkpoint(checkpoint)` by removing kwargs `model=`.')
        model = load_torch_model(saved_model=model_or_state_dict, model_definition=model)
        return model