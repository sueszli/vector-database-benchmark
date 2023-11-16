import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
import numpy as np
from typing import Dict
train_ds = ray.data.read_parquet('s3://anonymous@ray-example-data/iris.parquet')

def normalize_length(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if False:
        while True:
            i = 10
    new_col = batch['sepal.length'] / np.max(batch['sepal.length'])
    batch['normalized.sepal.length'] = new_col
    del batch['sepal.length']
    return batch
train_ds = train_ds.map_batches(normalize_length)

def train_loop_per_worker():
    if False:
        return 10
    it = train.get_dataset_shard('train')
    for _ in range(10):
        for batch in it.iter_batches(local_shuffle_buffer_size=10000, batch_size=128, prefetch_batches=10):
            print('Do some training on batch', batch)
my_trainer = TorchTrainer(train_loop_per_worker, scaling_config=ScalingConfig(num_workers=2), datasets={'train': train_ds})
my_trainer.fit()
dataset_a = ray.data.read_text('s3://anonymous@ray-example-data/sms_spam_collection_subset.txt')
dataset_b = ray.data.read_csv('s3://anonymous@ray-example-data/dow_jones.csv')
my_trainer = TorchTrainer(train_loop_per_worker, scaling_config=ScalingConfig(num_workers=2), datasets={'a': dataset_a, 'b': dataset_b}, dataset_config=ray.train.DataConfig(datasets_to_split=['a']))

def augment_data(batch):
    if False:
        while True:
            i = 10
    return batch
train_ds = ray.data.read_parquet('s3://anonymous@ray-example-data/iris.parquet')
train_ds = train_ds.map_batches(normalize_length)
train_ds = train_ds.materialize()
train_ds = train_ds.map_batches(augment_data)
from ray.train import DataConfig
options = DataConfig.default_ingest_options()
options.resource_limits.object_store_memory = 10000000000.0
my_trainer = TorchTrainer(train_loop_per_worker, scaling_config=ScalingConfig(num_workers=2), dataset_config=ray.train.DataConfig(execution_options=options))
from typing import Optional, Dict, List
from ray.data import Dataset, DataIterator, NodeIdStr
from ray.actor import ActorHandle

class MyCustomDataConfig(DataConfig):

    def configure(self, datasets: Dict[str, Dataset], world_size: int, worker_handles: Optional[List[ActorHandle]], worker_node_ids: Optional[List[NodeIdStr]], **kwargs) -> List[Dict[str, DataIterator]]:
        if False:
            for i in range(10):
                print('nop')
        assert len(datasets) == 1, 'This example only handles the simple case'
        ctx = ray.data.DataContext.get_current()
        ctx.execution_options = DataConfig.default_ingest_options()
        iterator_shards = datasets['train'].streaming_split(world_size, equal=True, locality_hints=worker_node_ids)
        return [{'train': it} for it in iterator_shards]
my_trainer = TorchTrainer(train_loop_per_worker, scaling_config=ScalingConfig(num_workers=2), datasets={'train': train_ds}, dataset_config=MyCustomDataConfig())
my_trainer.fit()