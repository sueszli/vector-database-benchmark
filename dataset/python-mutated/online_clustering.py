import datetime
from operator import itemgetter
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import apache_beam as beam
from apache_beam.coders import PickleCoder
from apache_beam.coders import VarIntCoder
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.sklearn_inference import ModelFileType
from apache_beam.ml.inference.sklearn_inference import SklearnModelHandlerNumpy
from apache_beam.transforms import core
from apache_beam.transforms import ptransform
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec

class SaveModel(core.DoFn):
    """Saves trained clustering model to persistent storage"""

    def __init__(self, checkpoints_path: str):
        if False:
            print('Hello World!')
        self.checkpoints_path = checkpoints_path

    def process(self, model):
        if False:
            return 10
        iso_timestamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        checkpoint_name = f'{self.checkpoints_path}/{iso_timestamp}.checkpoint'
        latest_checkpoint = f'{self.checkpoints_path}/latest.checkpoint'
        if FileSystems.exists(latest_checkpoint):
            FileSystems.rename([latest_checkpoint], [checkpoint_name])
        file = FileSystems.create(latest_checkpoint, 'wb')
        if not joblib:
            raise ImportError('Could not import joblib in this execution environment. For help with managing dependencies on Python workers.see https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/')
        joblib.dump(model, file)
        yield checkpoint_name

class AssignClusterLabelsFn(core.DoFn):
    """Takes a trained model and input data and labels
   all data instances using the trained model."""

    def process(self, batch, model, model_id):
        if False:
            for i in range(10):
                print('nop')
        cluster_labels = model.predict(batch)
        for (e, i) in zip(batch, cluster_labels):
            yield PredictionResult(example=e, inference=i, model_id=model_id)

class SelectLatestModelState(core.CombineFn):
    """Selects that latest version of a model after training"""

    def create_accumulator(self):
        if False:
            i = 10
            return i + 15
        return (None, 0)

    def add_input(self, accumulator, element):
        if False:
            for i in range(10):
                print('nop')
        if element[1] > accumulator[1]:
            return element
        return accumulator

    def merge_accumulators(self, accumulators):
        if False:
            for i in range(10):
                print('nop')
        return max(accumulators, key=itemgetter(1))

    def extract_output(self, accumulator):
        if False:
            return 10
        return accumulator[0]

class ClusteringAlgorithm(core.DoFn):
    """Abstract class with the interface
   that clustering algorithms need to follow."""
    MODEL_SPEC = ReadModifyWriteStateSpec('clustering_model', PickleCoder())
    ITERATION_SPEC = ReadModifyWriteStateSpec('training_iterations', VarIntCoder())
    MODEL_ID = 'ClusteringAlgorithm'

    def __init__(self, n_clusters: int, checkpoints_path: str, cluster_args: dict):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.n_clusters = n_clusters
        self.checkpoints_path = checkpoints_path
        self.cluster_args = cluster_args
        self.clustering_algorithm = None

    def process(self, keyed_batch, model_state=core.DoFn.StateParam(MODEL_SPEC), iteration_state=core.DoFn.StateParam(ITERATION_SPEC), *args, **kwargs):
        if False:
            return 10
        raise NotImplementedError

    def load_model_checkpoint(self):
        if False:
            i = 10
            return i + 15
        latest_checkpoint = f'{self.checkpoints_path}/latest.checkpoint'
        if FileSystems.exists(latest_checkpoint):
            file = FileSystems.open(latest_checkpoint, 'rb')
            if not joblib:
                raise ImportError('Could not import joblib in this execution environment. For help with managing dependencies on Python workers.see https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/')
            return joblib.load(file)
        return self.clustering_algorithm(n_clusters=self.n_clusters, **self.cluster_args)

class OnlineKMeans(ClusteringAlgorithm):
    """Online K-Means function. Used the MiniBatchKMeans from sklearn
    More information: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html"""
    MODEL_SPEC = ReadModifyWriteStateSpec('clustering_model', PickleCoder())
    ITERATION_SPEC = ReadModifyWriteStateSpec('training_iterations', VarIntCoder())
    MODEL_ID = 'OnlineKmeans'

    def __init__(self, n_clusters: int, checkpoints_path: str, cluster_args: dict):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(n_clusters, checkpoints_path, cluster_args)
        self.clustering_algorithm = MiniBatchKMeans

    def process(self, keyed_batch, model_state=core.DoFn.StateParam(MODEL_SPEC), iteration_state=core.DoFn.StateParam(ITERATION_SPEC), *args, **kwargs):
        if False:
            while True:
                i = 10
        clustering = model_state.read() or self.load_model_checkpoint()
        iteration = iteration_state.read() or 0
        iteration += 1
        (_, batch) = keyed_batch
        clustering.partial_fit(batch)
        model_state.write(clustering)
        iteration_state.write(iteration)
        yield (clustering, iteration)

class ConvertToNumpyArray(core.DoFn):
    """Helper function to convert incoming data
  to numpy arrays that are accepted by sklearn"""

    def process(self, element, *args, **kwargs):
        if False:
            print('Hello World!')
        if isinstance(element, (tuple, list)):
            yield np.array(element)
        elif isinstance(element, np.ndarray):
            yield element
        elif isinstance(element, (pd.DataFrame, pd.Series)):
            yield element.to_numpy()
        else:
            raise ValueError(f'Unsupported type: {type(element)}')

class ClusteringPreprocessing(ptransform.PTransform):

    def __init__(self, n_clusters: int, batch_size: int, is_batched: bool=False):
        if False:
            while True:
                i = 10
        " Preprocessing for Clustering Transformation\n        The clustering transform expects batches for performance reasons,\n        therefore this batches the data and converts it to numpy arrays,\n        which are accepted by sklearn. This transform also adds the same key\n        to all batches, such that only 1 state is created and updated during\n        clustering updates.\n\n          Example Usage::\n\n            pcoll | ClusteringPreprocessing(\n              n_clusters=8,\n              batch_size=1024,\n              is_batched=False)\n\n          Args:\n          n_clusters: number of clusters used by the algorithm\n          batch_size: size of the data batches\n          is_batched: boolean value that marks if the collection is already\n            batched and thus doesn't need to be batched by this transform\n          "
        super().__init__()
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.is_batched = is_batched

    def expand(self, pcoll):
        if False:
            return 10
        pcoll = pcoll | 'Convert element to numpy arrays' >> beam.ParDo(ConvertToNumpyArray())
        if not self.is_batched:
            pcoll = pcoll | 'Create batches of elements' >> beam.BatchElements(min_batch_size=self.n_clusters, max_batch_size=self.batch_size) | 'Covert to 2d numpy array' >> beam.Map(lambda record: np.array(record))
        return pcoll

class OnlineClustering(ptransform.PTransform):

    def __init__(self, clustering_algorithm, n_clusters: int, cluster_args: dict, checkpoints_path: str, batch_size: int=1024, is_batched: bool=False):
        if False:
            for i in range(10):
                print('nop')
        " Clustering transformation itself, it first preprocesses the data,\n        then it applies the clustering transformation step by step on each\n        of the batches.\n\n          Example Usage::\n\n            pcoll | OnlineClustering(\n                        clustering_algorithm=OnlineKMeansClustering\n                        batch_size=1024,\n                        n_clusters=6\n                        cluster_args={}))\n\n          Args:\n          clustering_algorithm: Clustering algorithm (DoFn)\n          n_clusters: Number of clusters\n          cluster_args: Arguments for the sklearn clustering algorithm\n            (check sklearn documentation for more information)\n          batch_size: size of the data batches\n          is_batched: boolean value that marks if the collection is already\n            batched and thus doesn't need to be batched by this transform\n          "
        super().__init__()
        self.clustering_algorithm = clustering_algorithm
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.cluster_args = cluster_args
        self.checkpoints_path = checkpoints_path
        self.is_batched = is_batched

    def expand(self, pcoll):
        if False:
            return 10
        data = pcoll | 'Batch data for faster processing' >> ClusteringPreprocessing(n_clusters=self.n_clusters, batch_size=self.batch_size, is_batched=self.is_batched) | 'Add a key for stateful processing' >> beam.Map(lambda record: (1, record))
        model = data | 'Cluster' >> core.ParDo(self.clustering_algorithm(n_clusters=self.n_clusters, cluster_args=self.cluster_args, checkpoints_path=self.checkpoints_path)) | 'Select latest model state' >> core.CombineGlobally(SelectLatestModelState()).without_defaults()
        _ = model | core.ParDo(SaveModel(checkpoints_path=self.checkpoints_path))
        return model

class AssignClusterLabelsRunInference(ptransform.PTransform):

    def __init__(self, checkpoints_path):
        if False:
            return 10
        super().__init__()
        self.clustering_model = SklearnModelHandlerNumpy(model_uri=f'{checkpoints_path}/latest.checkpoint', model_file_type=ModelFileType.JOBLIB)

    def expand(self, pcoll):
        if False:
            while True:
                i = 10
        predictions = pcoll | 'RunInference' >> RunInference(self.clustering_model)
        return predictions

class AssignClusterLabelsInMemoryModel(ptransform.PTransform):

    def __init__(self, model, n_clusters, batch_size, is_batched=False, model_id=None):
        if False:
            return 10
        self.model = model
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.is_batched = is_batched
        self.model_id = model_id

    def expand(self, pcoll):
        if False:
            while True:
                i = 10
        return pcoll | 'Preprocess data for faster prediction' >> ClusteringPreprocessing(n_clusters=self.n_clusters, batch_size=self.batch_size, is_batched=self.is_batched) | 'Assign cluster labels' >> core.ParDo(AssignClusterLabelsFn(), model=self.model, model_id=self.model_id)