from bigdl.orca.data import SparkXShards
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from bigdl.orca.data.tf.data import Dataset
    from bigdl.orca.data.shard import SparkXShards
    from bigdl.orca.data.ray_xshards import RayXShards

class TF2Dataset(object):

    def __init__(self, dataset: 'Dataset') -> None:
        if False:
            return 10
        self.rdd = dataset.as_tf_dataset_rdd()
        self.dataset = dataset

    def get_origin_xshards(self) -> 'SparkXShards':
        if False:
            i = 10
            return i + 15
        return self.dataset.get_xshards()

    def get_xshards(self) -> 'SparkXShards':
        if False:
            while True:
                i = 10
        return SparkXShards(self.rdd)

    def get_ray_xshards(self, num_workers: int) -> 'RayXShards':
        if False:
            print('Hello World!')
        from bigdl.orca.data.utils import process_spark_xshards
        xshards = self.get_xshards()
        return process_spark_xshards(xshards, num_workers)