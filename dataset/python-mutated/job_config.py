import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from ray.util.annotations import PublicAPI
if TYPE_CHECKING:
    from ray.runtime_env import RuntimeEnv

@PublicAPI
class JobConfig:
    """A class used to store the configurations of a job.

    Examples:
        .. testcode::
            :hide:

            import ray
            ray.shutdown()

        .. testcode::

            import ray
            from ray.job_config import JobConfig

            ray.init(job_config=JobConfig(default_actor_lifetime="non_detached"))

    Args:
        jvm_options: The jvm options for java workers of the job.
        code_search_path: A list of directories or jar files that
            specify the search path for user code. This will be used as
            `CLASSPATH` in Java and `PYTHONPATH` in Python.
            See :ref:`Ray cross-language programming <cross_language>` for more details.
        runtime_env: A :ref:`runtime environment <runtime-environments>` dictionary.
        metadata: An opaque metadata dictionary.
        ray_namespace: A :ref:`namespace <namespaces-guide>`
            is a logical grouping of jobs and named actors.
        default_actor_lifetime: The default value of actor lifetime,
            can be "detached" or "non_detached".
            See :ref:`actor lifetimes <actor-lifetimes>` for more details.
    """

    def __init__(self, jvm_options: Optional[List[str]]=None, code_search_path: Optional[List[str]]=None, runtime_env: Optional[dict]=None, _client_job: bool=False, metadata: Optional[dict]=None, ray_namespace: Optional[str]=None, default_actor_lifetime: str='non_detached', _py_driver_sys_path: Optional[List[str]]=None):
        if False:
            while True:
                i = 10
        self.jvm_options = jvm_options or []
        self.code_search_path = code_search_path or []
        assert isinstance(self.code_search_path, (list, tuple)), f'The type of code search path is incorrect: {type(code_search_path)}'
        self._client_job = _client_job
        self.metadata = metadata or {}
        self.ray_namespace = ray_namespace
        self.set_runtime_env(runtime_env)
        self.set_default_actor_lifetime(default_actor_lifetime)
        self._py_driver_sys_path = _py_driver_sys_path or []

    def set_metadata(self, key: str, value: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Add key-value pair to the metadata dictionary.\n\n        If the key already exists, the value is overwritten to the new value.\n\n        Examples:\n            .. testcode::\n\n                import ray\n                from ray.job_config import JobConfig\n\n                job_config = JobConfig()\n                job_config.set_metadata("submitter", "foo")\n\n        Args:\n            key: The key of the metadata.\n            value: The value of the metadata.\n        '
        self.metadata[key] = value

    def _serialize(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Serialize the struct into protobuf string'
        return self._get_proto_job_config().SerializeToString()

    def set_runtime_env(self, runtime_env: Optional[Union[Dict[str, Any], 'RuntimeEnv']], validate: bool=False) -> None:
        if False:
            print('Hello World!')
        "Modify the runtime_env of the JobConfig.\n\n        We don't validate the runtime_env by default here because it may go\n        through some translation before actually being passed to C++ (e.g.,\n        working_dir translated from a local directory to a URI).\n\n        Args:\n            runtime_env: A :ref:`runtime environment <runtime-environments>` dictionary.\n            validate: Whether to validate the runtime env.\n        "
        self.runtime_env = runtime_env if runtime_env is not None else {}
        if validate:
            self.runtime_env = self._validate_runtime_env()
        self._cached_pb = None

    def set_ray_namespace(self, ray_namespace: str) -> None:
        if False:
            i = 10
            return i + 15
        'Set Ray :ref:`namespace <namespaces-guide>`.\n\n        Args:\n            ray_namespace: The namespace to set.\n        '
        if ray_namespace != self.ray_namespace:
            self.ray_namespace = ray_namespace
            self._cached_pb = None

    def set_default_actor_lifetime(self, default_actor_lifetime: str) -> None:
        if False:
            i = 10
            return i + 15
        'Set the default actor lifetime, which can be "detached" or "non_detached".\n\n        See :ref:`actor lifetimes <actor-lifetimes>` for more details.\n\n        Args:\n            default_actor_lifetime: The default actor lifetime to set.\n        '
        import ray.core.generated.common_pb2 as common_pb2
        if default_actor_lifetime == 'detached':
            self._default_actor_lifetime = common_pb2.JobConfig.ActorLifetime.DETACHED
        elif default_actor_lifetime == 'non_detached':
            self._default_actor_lifetime = common_pb2.JobConfig.ActorLifetime.NON_DETACHED
        else:
            raise ValueError('Default actor lifetime must be one of `detached`, `non_detached`')

    def _validate_runtime_env(self):
        if False:
            for i in range(10):
                print('nop')
        from ray.runtime_env import RuntimeEnv
        if isinstance(self.runtime_env, RuntimeEnv):
            return self.runtime_env
        return RuntimeEnv(**self.runtime_env)

    def _get_proto_job_config(self):
        if False:
            i = 10
            return i + 15
        'Return the protobuf structure of JobConfig.'
        import ray.core.generated.common_pb2 as common_pb2
        from ray._private.utils import get_runtime_env_info
        if self._cached_pb is None:
            pb = common_pb2.JobConfig()
            if self.ray_namespace is None:
                pb.ray_namespace = str(uuid.uuid4())
            else:
                pb.ray_namespace = self.ray_namespace
            pb.jvm_options.extend(self.jvm_options)
            pb.code_search_path.extend(self.code_search_path)
            pb.py_driver_sys_path.extend(self._py_driver_sys_path)
            for (k, v) in self.metadata.items():
                pb.metadata[k] = v
            parsed_env = self._validate_runtime_env()
            pb.runtime_env_info.CopyFrom(get_runtime_env_info(parsed_env, is_job_runtime_env=True, serialize=False))
            if self._default_actor_lifetime is not None:
                pb.default_actor_lifetime = self._default_actor_lifetime
            self._cached_pb = pb
        return self._cached_pb

    def _runtime_env_has_working_dir(self):
        if False:
            for i in range(10):
                print('nop')
        return self._validate_runtime_env().has_working_dir()

    def _get_serialized_runtime_env(self) -> str:
        if False:
            return 10
        'Return the JSON-serialized parsed runtime env dict'
        return self._validate_runtime_env().serialize()

    def _get_proto_runtime_env_config(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return the JSON-serialized parsed runtime env info'
        return self._get_proto_job_config().runtime_env_info.runtime_env_config

    @classmethod
    def from_json(cls, job_config_json):
        if False:
            for i in range(10):
                print('nop')
        'Generates a JobConfig object from json.\n\n        Examples:\n            .. testcode::\n\n                from ray.job_config import JobConfig\n\n                job_config = JobConfig.from_json(\n                    {"runtime_env": {"working_dir": "uri://abc"}})\n\n        Args:\n            job_config_json: The job config json dictionary.\n        '
        return cls(jvm_options=job_config_json.get('jvm_options', None), code_search_path=job_config_json.get('code_search_path', None), runtime_env=job_config_json.get('runtime_env', None), metadata=job_config_json.get('metadata', None), ray_namespace=job_config_json.get('ray_namespace', None), _client_job=job_config_json.get('client_job', False), _py_driver_sys_path=job_config_json.get('py_driver_sys_path', None))