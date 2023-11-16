"""A configure tuple for high-level APIs for running distribution strategies."""
import collections

class DistributeConfig(collections.namedtuple('DistributeConfig', ['train_distribute', 'eval_distribute', 'remote_cluster'])):
    """A config tuple for distribution strategies.

  Attributes:
    train_distribute: a `DistributionStrategy` object for training.
    eval_distribute: an optional `DistributionStrategy` object for
      evaluation.
    remote_cluster: a dict, `ClusterDef` or `ClusterSpec` object specifying
      the cluster configurations. If this is given, the `train_and_evaluate`
      method will be running as a standalone client which connects to the
      cluster for training.
  """

    def __new__(cls, train_distribute=None, eval_distribute=None, remote_cluster=None):
        if False:
            i = 10
            return i + 15
        return super(DistributeConfig, cls).__new__(cls, train_distribute, eval_distribute, remote_cluster)