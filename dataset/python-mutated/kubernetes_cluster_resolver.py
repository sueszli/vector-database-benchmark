"""Implementation of Cluster Resolvers for Kubernetes."""
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training import server_lib
from tensorflow.python.util.tf_export import tf_export

@tf_export('distribute.cluster_resolver.KubernetesClusterResolver')
class KubernetesClusterResolver(ClusterResolver):
    """ClusterResolver for Kubernetes.

  This is an implementation of cluster resolvers for Kubernetes. When given the
  the Kubernetes namespace and label selector for pods, we will retrieve the
  pod IP addresses of all running pods matching the selector, and return a
  ClusterSpec based on that information.

  Note: it cannot retrieve `task_type`, `task_id` or `rpc_layer`. To use it
  with some distribution strategies like
  `tf.distribute.experimental.MultiWorkerMirroredStrategy`, you will need to
  specify `task_type` and `task_id` by setting these attributes.

  Usage example with tf.distribute.Strategy:

    ```Python
    # On worker 0
    cluster_resolver = KubernetesClusterResolver(
        {"worker": ["job-name=worker-cluster-a", "job-name=worker-cluster-b"]})
    cluster_resolver.task_type = "worker"
    cluster_resolver.task_id = 0
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver)

    # On worker 1
    cluster_resolver = KubernetesClusterResolver(
        {"worker": ["job-name=worker-cluster-a", "job-name=worker-cluster-b"]})
    cluster_resolver.task_type = "worker"
    cluster_resolver.task_id = 1
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver)
    ```
  """

    def __init__(self, job_to_label_mapping=None, tf_server_port=8470, rpc_layer='grpc', override_client=None):
        if False:
            for i in range(10):
                print('nop')
        "Initializes a new KubernetesClusterResolver.\n\n    This initializes a new Kubernetes ClusterResolver. The ClusterResolver\n    will attempt to talk to the Kubernetes master to retrieve all the instances\n    of pods matching a label selector.\n\n    Args:\n      job_to_label_mapping: A mapping of TensorFlow jobs to label selectors.\n        This allows users to specify many TensorFlow jobs in one Cluster\n        Resolver, and each job can have pods belong with different label\n        selectors. For example, a sample mapping might be\n        ```\n        {'worker': ['job-name=worker-cluster-a', 'job-name=worker-cluster-b'],\n         'ps': ['job-name=ps-1', 'job-name=ps-2']}\n        ```\n      tf_server_port: The port the TensorFlow server is listening on.\n      rpc_layer: (Optional) The RPC layer TensorFlow should use to communicate\n        between tasks in Kubernetes. Defaults to 'grpc'.\n      override_client: The Kubernetes client (usually automatically retrieved\n        using `from kubernetes import client as k8sclient`). If you pass this\n        in, you are responsible for setting Kubernetes credentials manually.\n\n    Raises:\n      ImportError: If the Kubernetes Python client is not installed and no\n        `override_client` is passed in.\n      RuntimeError: If autoresolve_task is not a boolean or a callable.\n    "
        try:
            from kubernetes import config as k8sconfig
            k8sconfig.load_kube_config()
        except ImportError:
            if not override_client:
                raise ImportError('The Kubernetes Python client must be installed before using the Kubernetes Cluster Resolver. To install the Kubernetes Python client, run `pip install kubernetes` on your command line.')
        if not job_to_label_mapping:
            job_to_label_mapping = {'worker': ['job-name=tensorflow']}
        self._job_to_label_mapping = job_to_label_mapping
        self._tf_server_port = tf_server_port
        self._override_client = override_client
        self.task_type = None
        self.task_id = None
        self.rpc_layer = rpc_layer

    def master(self, task_type=None, task_id=None, rpc_layer=None):
        if False:
            print('Hello World!')
        'Returns the master address to use when creating a session.\n\n    You must have set the task_type and task_id object properties before\n    calling this function, or pass in the `task_type` and `task_id`\n    parameters when using this function. If you do both, the function parameters\n    will override the object properties.\n\n    Note: this is only useful for TensorFlow 1.x.\n\n    Args:\n      task_type: (Optional) The type of the TensorFlow task of the master.\n      task_id: (Optional) The index of the TensorFlow task of the master.\n      rpc_layer: (Optional) The RPC protocol for the given cluster.\n\n    Returns:\n      The name or URL of the session master.\n    '
        task_type = task_type if task_type is not None else self.task_type
        task_id = task_id if task_id is not None else self.task_id
        if task_type is not None and task_id is not None:
            return format_master_url(self.cluster_spec().task_address(task_type, task_id), rpc_layer or self.rpc_layer)
        return ''

    def cluster_spec(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a ClusterSpec object based on the latest info from Kubernetes.\n\n    We retrieve the information from the Kubernetes master every time this\n    method is called.\n\n    Returns:\n      A ClusterSpec containing host information returned from Kubernetes.\n\n    Raises:\n      RuntimeError: If any of the pods returned by the master is not in the\n        `Running` phase.\n    '
        if self._override_client:
            client = self._override_client
        else:
            from kubernetes import config as k8sconfig
            from kubernetes import client as k8sclient
            k8sconfig.load_kube_config()
            client = k8sclient.CoreV1Api()
        cluster_map = {}
        for tf_job in self._job_to_label_mapping:
            all_pods = []
            for selector in self._job_to_label_mapping[tf_job]:
                ret = client.list_pod_for_all_namespaces(label_selector=selector)
                selected_pods = []
                for pod in sorted(ret.items, key=lambda x: x.metadata.name):
                    if pod.status.phase == 'Running':
                        selected_pods.append('%s:%s' % (pod.status.host_ip, self._tf_server_port))
                    else:
                        raise RuntimeError('Pod "%s" is not running; phase: "%s"' % (pod.metadata.name, pod.status.phase))
                all_pods.extend(selected_pods)
            cluster_map[tf_job] = all_pods
        return server_lib.ClusterSpec(cluster_map)