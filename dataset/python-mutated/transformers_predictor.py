import logging
from typing import TYPE_CHECKING, List, Optional, Type, Union
import pandas as pd
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.air.data_batch_type import DataBatchType
from ray.train.predictor import Predictor
from ray.util import log_once
from ray.util.annotations import Deprecated
try:
    import torch
    torch_get_gpus = torch.cuda.device_count
except ImportError:

    def torch_get_gpus():
        if False:
            i = 10
            return i + 15
        return 0
try:
    import tensorflow

    def tf_get_gpus():
        if False:
            print('Hello World!')
        return len(tensorflow.config.list_physical_devices('GPU'))
except ImportError:

    def tf_get_gpus():
        if False:
            for i in range(10):
                print('nop')
        return 0
TRANSFORMERS_IMPORT_ERROR: Optional[ImportError] = None
try:
    from transformers.pipelines import Pipeline
    from transformers.pipelines import pipeline as pipeline_factory
    from transformers.pipelines.table_question_answering import TableQuestionAnsweringPipeline
except ImportError as e:
    TRANSFORMERS_IMPORT_ERROR = e
if TYPE_CHECKING:
    from transformers.modeling_tf_utils import TFPreTrainedModel
    from transformers.modeling_utils import PreTrainedModel
    from ray.data.preprocessor import Preprocessor
    from ray.train.huggingface import TransformersCheckpoint
logger = logging.getLogger(__name__)
TRANSFORMERS_PREDICTOR_DEPRECATION_MESSAGE = 'The TransformersPredictor will be hard deprecated in Ray 2.8. Use TorchTrainer instead. For batch inference, see https://docs.ray.io/en/master/data/batch_inference.htmlfor more details.'

@Deprecated
class TransformersPredictor(Predictor):
    """A predictor for HuggingFace Transformers PyTorch models.

    This predictor uses Transformers Pipelines for inference.

    Args:
        pipeline: The Transformers pipeline to use for inference.
        preprocessor: A preprocessor used to transform data batches prior
            to prediction.
        use_gpu: If set, the model will be moved to GPU on instantiation and
            prediction happens on GPU.
    """

    def __init__(self, pipeline: Optional['Pipeline']=None, preprocessor: Optional['Preprocessor']=None, use_gpu: bool=False):
        if False:
            while True:
                i = 10
        raise DeprecationWarning(TRANSFORMERS_PREDICTOR_DEPRECATION_MESSAGE)
        if TRANSFORMERS_IMPORT_ERROR is not None:
            raise TRANSFORMERS_IMPORT_ERROR
        self.pipeline = pipeline
        self.use_gpu = use_gpu
        num_gpus = max(torch_get_gpus(), tf_get_gpus())
        if not use_gpu and num_gpus > 0 and log_once('hf_predictor_not_using_gpu'):
            logger.warning(f'You have `use_gpu` as False but there are {num_gpus} GPUs detected on host where prediction will only use CPU. Please consider explicitly setting `TransformersPredictor(use_gpu=True)` or `batch_predictor.predict(ds, num_gpus_per_worker=1)` to enable GPU prediction. Ignore if you have set `device` or `device_map` arguments in the `pipeline` manually.')
        super().__init__(preprocessor)

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'{self.__class__.__name__}(pipeline={self.pipeline!r}, preprocessor={self._preprocessor!r})'

    @classmethod
    def from_checkpoint(cls, checkpoint: 'TransformersCheckpoint', *, pipeline_cls: Optional[Type['Pipeline']]=None, model_cls: Optional[Union[str, Type['PreTrainedModel'], Type['TFPreTrainedModel']]]=None, pretrained_model_kwargs: Optional[dict]=None, use_gpu: bool=False, **pipeline_kwargs) -> 'TransformersPredictor':
        if False:
            return 10
        "Instantiate the predictor from a TransformersCheckpoint.\n\n        Note that the Transformers ``pipeline`` used internally expects to\n        receive raw text. If you have any Preprocessors in Checkpoint\n        that tokenize the data, remove them by calling\n        ``Checkpoint.set_preprocessor(None)`` beforehand.\n\n        Args:\n            checkpoint: The checkpoint to load the model, tokenizer and\n                preprocessor from.\n            pipeline_cls: A ``transformers.pipelines.Pipeline`` class to use.\n                If not specified, will use the ``pipeline`` abstraction\n                wrapper.\n            model_cls: A ``transformers.PreTrainedModel`` class to create from\n                the checkpoint.\n            pretrained_model_kwargs: If set and a ``model_cls`` is provided, will\n                be passed to ``TransformersCheckpoint.get_model()``.\n            use_gpu: If set, the model will be moved to GPU on instantiation and\n                prediction happens on GPU.\n            **pipeline_kwargs: Any kwargs to pass to the pipeline\n                initialization. If ``pipeline_cls`` is None, this must contain\n                the 'task' argument. Can be used\n                to override the tokenizer with 'tokenizer'. If ``use_gpu`` is\n                True, 'device' will be set to 0 by default, unless 'device_map' is\n                passed.\n        "
        if TRANSFORMERS_IMPORT_ERROR is not None:
            raise TRANSFORMERS_IMPORT_ERROR
        if not pipeline_cls and 'task' not in pipeline_kwargs:
            raise ValueError("If `pipeline_cls` is not specified, 'task' must be passed as a kwarg.")
        if use_gpu and 'device_map' not in pipeline_kwargs:
            pipeline_kwargs.setdefault('device', 0)
        model = None
        if model_cls:
            pretrained_model_kwargs = pretrained_model_kwargs or {}
            model = checkpoint.get_model(model_cls, **pretrained_model_kwargs)
        if pipeline_cls and model:
            pipeline = pipeline_cls(model, **pipeline_kwargs)
        else:
            if pipeline_cls:
                pipeline_kwargs['pipeline_class'] = pipeline_cls
            if not model:
                with checkpoint.as_directory() as checkpoint_path:
                    pipeline = pipeline_factory(model=checkpoint_path, **pipeline_kwargs)
            else:
                pipeline = pipeline_factory(model=model, **pipeline_kwargs)
        preprocessor = checkpoint.get_preprocessor()
        return cls(pipeline=pipeline, preprocessor=preprocessor, use_gpu=use_gpu)

    def _predict(self, data: Union[list, pd.DataFrame], **pipeline_call_kwargs) -> pd.DataFrame:
        if False:
            i = 10
            return i + 15
        ret = self.pipeline(data, **pipeline_call_kwargs)
        try:
            new_ret = [x[0] if isinstance(x, list) and len(x) == 1 else x for x in ret]
            df = pd.DataFrame(new_ret)
        except Exception:
            df = pd.DataFrame(ret)
        df.columns = [str(col) for col in df.columns]
        return df

    @staticmethod
    def _convert_data_for_pipeline(data: pd.DataFrame, pipeline: 'Pipeline') -> Union[list, pd.DataFrame]:
        if False:
            while True:
                i = 10
        'Convert the data into a format accepted by the pipeline.\n\n        In most cases, this format is a list of strings.'
        if isinstance(pipeline, TableQuestionAnsweringPipeline):
            return data
        columns = [data[col].to_list() for col in data.columns]
        while isinstance(columns, list) and len(columns) == 1:
            columns = columns[0]
        return columns

    def predict(self, data: DataBatchType, feature_columns: Optional[Union[List[str], List[int]]]=None, **predict_kwargs) -> DataBatchType:
        if False:
            print('Hello World!')
        'Run inference on data batch.\n\n        The data is converted into a list (unless ``pipeline`` is a\n        ``TableQuestionAnsweringPipeline``) and passed to the ``pipeline``\n        object.\n\n        Args:\n            data: A batch of input data. Either a pandas DataFrame or numpy\n                array.\n            feature_columns: The names or indices of the columns in the\n                data to use as features to predict on. If None, use all\n                columns.\n            **pipeline_call_kwargs: additional kwargs to pass to the\n                ``pipeline`` object.\n\n        Returns:\n            Prediction result.\n        '
        return Predictor.predict(self, data, feature_columns=feature_columns, **predict_kwargs)

    def _predict_pandas(self, data: 'pd.DataFrame', feature_columns: Optional[List[str]]=None, **pipeline_call_kwargs) -> 'pd.DataFrame':
        if False:
            i = 10
            return i + 15
        if TENSOR_COLUMN_NAME in data:
            arr = data[TENSOR_COLUMN_NAME].to_numpy()
            if feature_columns:
                data = pd.DataFrame(arr[:, feature_columns])
        elif feature_columns:
            data = data[feature_columns]
        data = data[feature_columns] if feature_columns else data
        data = self._convert_data_for_pipeline(data, self.pipeline)
        return self._predict(data, **pipeline_call_kwargs)