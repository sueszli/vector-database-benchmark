import logging
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type, Union
import numpy as np
import tensorflow as tf
from ray.air._internal.tensorflow_utils import convert_ndarray_batch_to_tf_tensor_batch
from ray.train._internal.dl_predictor import DLPredictor
from ray.train.predictor import DataBatchType
from ray.train.tensorflow import TensorflowCheckpoint
from ray.util import log_once
from ray.util.annotations import DeveloperAPI, PublicAPI
if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor
logger = logging.getLogger(__name__)

@PublicAPI(stability='beta')
class TensorflowPredictor(DLPredictor):
    """A predictor for TensorFlow models.

    Args:
        model: A Tensorflow Keras model to use for predictions.
        preprocessor: A preprocessor used to transform data batches prior
            to prediction.
        model_weights: List of weights to use for the model.
        use_gpu: If set, the model will be moved to GPU on instantiation and
            prediction happens on GPU.
    """

    def __init__(self, *, model: Optional[tf.keras.Model]=None, preprocessor: Optional['Preprocessor']=None, use_gpu: bool=False):
        if False:
            i = 10
            return i + 15
        self.use_gpu = use_gpu
        if use_gpu:
            with tf.device('GPU:0'):
                self._model = model
        else:
            self._model = model
            gpu_devices = tf.config.list_physical_devices('GPU')
            if len(gpu_devices) > 0 and log_once('tf_predictor_not_using_gpu'):
                logger.warning(f'You have `use_gpu` as False but there are {len(gpu_devices)} GPUs detected on host where prediction will only use CPU. Please consider explicitly setting `TensorflowPredictor(use_gpu=True)` or `batch_predictor.predict(ds, num_gpus_per_worker=1)` to enable GPU prediction.')
        super().__init__(preprocessor)

    def __repr__(self):
        if False:
            print('Hello World!')
        fn_name = getattr(self._model, '__name__', self._model)
        fn_name_str = ''
        if fn_name:
            fn_name_str = str(fn_name)[:40]
        return f'{self.__class__.__name__}(model={fn_name_str!r}, preprocessor={self._preprocessor!r}, use_gpu={self.use_gpu!r})'

    @classmethod
    def from_checkpoint(cls, checkpoint: TensorflowCheckpoint, model_definition: Optional[Union[Callable[[], tf.keras.Model], Type[tf.keras.Model]]]=None, use_gpu: Optional[bool]=False) -> 'TensorflowPredictor':
        if False:
            i = 10
            return i + 15
        'Instantiate the predictor from a TensorflowCheckpoint.\n\n        Args:\n            checkpoint: The checkpoint to load the model and preprocessor from.\n            model_definition: A callable that returns a TensorFlow Keras model\n                to use. Model weights will be loaded from the checkpoint.\n                This is only needed if the `checkpoint` was created from\n                `TensorflowCheckpoint.from_model`.\n            use_gpu: Whether GPU should be used during prediction.\n        '
        if model_definition:
            raise DeprecationWarning('`model_definition` is deprecated. `TensorflowCheckpoint.from_model` now saves the full model definition in .keras format.')
        model = checkpoint.get_model()
        preprocessor = checkpoint.get_preprocessor()
        return cls(model=model, preprocessor=preprocessor, use_gpu=use_gpu)

    @DeveloperAPI
    def call_model(self, inputs: Union[tf.Tensor, Dict[str, tf.Tensor]]) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        if False:
            i = 10
            return i + 15
        'Runs inference on a single batch of tensor data.\n\n        This method is called by `TorchPredictor.predict` after converting the\n        original data batch to torch tensors.\n\n        Override this method to add custom logic for processing the model input or\n        output.\n\n        Example:\n\n            .. testcode::\n\n                # List outputs are not supported by default TensorflowPredictor.\n                def build_model() -> tf.keras.Model:\n                    input = tf.keras.layers.Input(shape=1)\n                    model = tf.keras.models.Model(inputs=input, outputs=[input, input])\n                    return model\n\n                # Use a custom predictor to format model output as a dict.\n                class CustomPredictor(TensorflowPredictor):\n                    def call_model(self, inputs):\n                        model_output = super().call_model(inputs)\n                        return {\n                            str(i): model_output[i] for i in range(len(model_output))\n                        }\n\n                import numpy as np\n                data_batch = np.array([[0.5], [0.6], [0.7]], dtype=np.float32)\n\n                predictor = CustomPredictor(model=build_model())\n                predictions = predictor.predict(data_batch)\n\n        Args:\n            inputs: A batch of data to predict on, represented as either a single\n                TensorFlow tensor or for multi-input models, a dictionary of tensors.\n\n        Returns:\n            The model outputs, either as a single tensor or a dictionary of tensors.\n\n        '
        if self.use_gpu:
            with tf.device('GPU:0'):
                return self._model(inputs)
        else:
            return self._model(inputs)

    def predict(self, data: DataBatchType, dtype: Optional[Union[tf.dtypes.DType, Dict[str, tf.dtypes.DType]]]=None) -> DataBatchType:
        if False:
            i = 10
            return i + 15
        'Run inference on data batch.\n\n        If the provided data is a single array or a dataframe/table with a single\n        column, it will be converted into a single Tensorflow tensor before being\n        inputted to the model.\n\n        If the provided data is a multi-column table or a dict of numpy arrays,\n        it will be converted into a dict of tensors before being inputted to the\n        model. This is useful for multi-modal inputs (for example your model accepts\n        both image and text).\n\n        Args:\n            data: A batch of input data. Either a pandas DataFrame or numpy\n                array.\n            dtype: The dtypes to use for the tensors. Either a single dtype for all\n                tensors or a mapping from column name to dtype.\n\n        Examples:\n\n        .. testcode::\n\n            import numpy as np\n            import tensorflow as tf\n            from ray.train.tensorflow import TensorflowPredictor\n\n            def build_model():\n                return tf.keras.Sequential(\n                    [\n                        tf.keras.layers.InputLayer(input_shape=()),\n                        tf.keras.layers.Flatten(),\n                        tf.keras.layers.Dense(1),\n                    ]\n                )\n\n            weights = [np.array([[2.0]]), np.array([0.0])]\n            predictor = TensorflowPredictor(model=build_model())\n\n            data = np.asarray([1, 2, 3])\n            predictions = predictor.predict(data)\n\n            import pandas as pd\n            import tensorflow as tf\n            from ray.train.tensorflow import TensorflowPredictor\n\n            def build_model():\n                input1 = tf.keras.layers.Input(shape=(1,), name="A")\n                input2 = tf.keras.layers.Input(shape=(1,), name="B")\n                merged = tf.keras.layers.Concatenate(axis=1)([input1, input2])\n                output = tf.keras.layers.Dense(2, input_dim=2)(merged)\n                return tf.keras.models.Model(\n                    inputs=[input1, input2], outputs=output)\n\n            predictor = TensorflowPredictor(model=build_model())\n\n            # Pandas dataframe.\n            data = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])\n\n            predictions = predictor.predict(data)\n\n        Returns:\n            DataBatchType: Prediction result. The return type will be the same as the\n                input type.\n        '
        return super(TensorflowPredictor, self).predict(data=data, dtype=dtype)

    def _arrays_to_tensors(self, numpy_arrays: Union[np.ndarray, Dict[str, np.ndarray]], dtype: Optional[Union[tf.dtypes.DType, Dict[str, tf.dtypes.DType]]]) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        return convert_ndarray_batch_to_tf_tensor_batch(numpy_arrays, dtypes=dtype)

    def _tensor_to_array(self, tensor: tf.Tensor) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(tensor, tf.Tensor):
            raise ValueError(f'Expected the model to return either a tf.Tensor or a dict of tf.Tensor, but got {type(tensor)} instead. To support models with different output types, subclass TensorflowPredictor and override the `call_model` method to process the output into either torch.Tensor or Dict[str, torch.Tensor].')
        return tensor.numpy()