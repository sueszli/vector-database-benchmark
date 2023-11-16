import logging
from typing import TYPE_CHECKING, Dict, Optional, Union
import numpy as np
import torch
from ray.air._internal.torch_utils import convert_ndarray_batch_to_torch_tensor_batch
from ray.train._internal.dl_predictor import DLPredictor
from ray.train.predictor import DataBatchType
from ray.train.torch import TorchCheckpoint
from ray.util import log_once
from ray.util.annotations import DeveloperAPI, PublicAPI
if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor
logger = logging.getLogger(__name__)

@PublicAPI(stability='beta')
class TorchPredictor(DLPredictor):
    """A predictor for PyTorch models.

    Args:
        model: The torch module to use for predictions.
        preprocessor: A preprocessor used to transform data batches prior
            to prediction.
        use_gpu: If set, the model will be moved to GPU on instantiation and
            prediction happens on GPU.
    """

    def __init__(self, model: torch.nn.Module, preprocessor: Optional['Preprocessor']=None, use_gpu: bool=False):
        if False:
            return 10
        self.model = model
        self.model.eval()
        self.use_gpu = use_gpu
        if use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        if not use_gpu and torch.cuda.device_count() > 0 and log_once('torch_predictor_not_using_gpu'):
            logger.warning(f'You have `use_gpu` as False but there are {torch.cuda.device_count()} GPUs detected on host where prediction will only use CPU. Please consider explicitly setting `TorchPredictor(use_gpu=True)` or `batch_predictor.predict(ds, num_gpus_per_worker=1)` to enable GPU prediction.')
        super().__init__(preprocessor)

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'{self.__class__.__name__}(model={self.model!r}, preprocessor={self._preprocessor!r}, use_gpu={self.use_gpu!r})'

    @classmethod
    def from_checkpoint(cls, checkpoint: TorchCheckpoint, model: Optional[torch.nn.Module]=None, use_gpu: bool=False) -> 'TorchPredictor':
        if False:
            for i in range(10):
                print('nop')
        'Instantiate the predictor from a TorchCheckpoint.\n\n        Args:\n            checkpoint: The checkpoint to load the model and preprocessor from.\n            model: If the checkpoint contains a model state dict, and not\n                the model itself, then the state dict will be loaded to this\n                ``model``. If the checkpoint already contains the model itself,\n                this model argument will be discarded.\n            use_gpu: If set, the model will be moved to GPU on instantiation and\n                prediction happens on GPU.\n        '
        model = checkpoint.get_model(model)
        preprocessor = checkpoint.get_preprocessor()
        return cls(model=model, preprocessor=preprocessor, use_gpu=use_gpu)

    @DeveloperAPI
    def call_model(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if False:
            i = 10
            return i + 15
        'Runs inference on a single batch of tensor data.\n\n        This method is called by `TorchPredictor.predict` after converting the\n        original data batch to torch tensors.\n\n        Override this method to add custom logic for processing the model input or\n        output.\n\n        Args:\n            inputs: A batch of data to predict on, represented as either a single\n                PyTorch tensor or for multi-input models, a dictionary of tensors.\n\n        Returns:\n            The model outputs, either as a single tensor or a dictionary of tensors.\n\n        Example:\n\n            .. testcode::\n\n                import numpy as np\n                import torch\n                from ray.train.torch import TorchPredictor\n\n                # List outputs are not supported by default TorchPredictor.\n                # So let\'s define a custom TorchPredictor and override call_model\n                class MyModel(torch.nn.Module):\n                    def forward(self, input_tensor):\n                        return [input_tensor, input_tensor]\n\n                # Use a custom predictor to format model output as a dict.\n                class CustomPredictor(TorchPredictor):\n                    def call_model(self, inputs):\n                        model_output = super().call_model(inputs)\n                        return {\n                            str(i): model_output[i] for i in range(len(model_output))\n                        }\n\n                # create our data batch\n                data_batch = np.array([1, 2])\n                # create custom predictor and predict\n                predictor = CustomPredictor(model=MyModel())\n                predictions = predictor.predict(data_batch)\n                print(f"Predictions: {predictions.get(\'0\')}, {predictions.get(\'1\')}")\n\n            .. testoutput::\n\n                Predictions: [1 2], [1 2]\n\n        '
        with torch.no_grad():
            output = self.model(inputs)
        return output

    def predict(self, data: DataBatchType, dtype: Optional[Union[torch.dtype, Dict[str, torch.dtype]]]=None) -> DataBatchType:
        if False:
            i = 10
            return i + 15
        'Run inference on data batch.\n\n        If the provided data is a single array or a dataframe/table with a single\n        column, it will be converted into a single PyTorch tensor before being\n        inputted to the model.\n\n        If the provided data is a multi-column table or a dict of numpy arrays,\n        it will be converted into a dict of tensors before being inputted to the\n        model. This is useful for multi-modal inputs (for example your model accepts\n        both image and text).\n\n        Args:\n            data: A batch of input data of ``DataBatchType``.\n            dtype: The dtypes to use for the tensors. Either a single dtype for all\n                tensors or a mapping from column name to dtype.\n\n        Returns:\n            DataBatchType: Prediction result. The return type will be the same as the\n                input type.\n\n        Example:\n\n            .. testcode::\n\n                    import numpy as np\n                    import pandas as pd\n                    import torch\n                    import ray\n                    from ray.train.torch import TorchPredictor\n\n                    # Define a custom PyTorch module\n                    class CustomModule(torch.nn.Module):\n                        def __init__(self):\n                            super().__init__()\n                            self.linear1 = torch.nn.Linear(1, 1)\n                            self.linear2 = torch.nn.Linear(1, 1)\n\n                        def forward(self, input_dict: dict):\n                            out1 = self.linear1(input_dict["A"].unsqueeze(1))\n                            out2 = self.linear2(input_dict["B"].unsqueeze(1))\n                            return out1 + out2\n\n                    # Set manul seed so we get consistent output\n                    torch.manual_seed(42)\n\n                    # Use Standard PyTorch model\n                    model = torch.nn.Linear(2, 1)\n                    predictor = TorchPredictor(model=model)\n                    # Define our data\n                    data = np.array([[1, 2], [3, 4]])\n                    predictions = predictor.predict(data, dtype=torch.float)\n                    print(f"Standard model predictions: {predictions}")\n                    print("---")\n\n                    # Use Custom PyTorch model with TorchPredictor\n                    predictor = TorchPredictor(model=CustomModule())\n                    # Define our data and predict Customer model with TorchPredictor\n                    data = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])\n                    predictions = predictor.predict(data, dtype=torch.float)\n                    print(f"Custom model predictions: {predictions}")\n\n            .. testoutput::\n\n                Standard model predictions: {\'predictions\': array([[1.5487633],\n                       [3.8037925]], dtype=float32)}\n                ---\n                Custom model predictions:     predictions\n                0  [0.61623406]\n                1    [2.857038]\n        '
        return super(TorchPredictor, self).predict(data=data, dtype=dtype)

    def _arrays_to_tensors(self, numpy_arrays: Union[np.ndarray, Dict[str, np.ndarray]], dtype: Optional[Union[torch.dtype, Dict[str, torch.dtype]]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if False:
            print('Hello World!')
        return convert_ndarray_batch_to_torch_tensor_batch(numpy_arrays, dtypes=dtype, device=self.device)

    def _tensor_to_array(self, tensor: torch.Tensor) -> np.ndarray:
        if False:
            while True:
                i = 10
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f'Expected the model to return either a torch.Tensor or a dict of torch.Tensor, but got {type(tensor)} instead. To support models with different output types, subclass TorchPredictor and override the `call_model` method to process the output into either torch.Tensor or Dict[str, torch.Tensor].')
        return tensor.cpu().detach().numpy()