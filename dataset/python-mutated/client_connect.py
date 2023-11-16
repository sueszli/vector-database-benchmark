from typing import Any, Dict, List, Optional, Tuple
import logging
from ray._private.client_mode_hook import _explicitly_enable_client_mode, _set_client_hook_status
from ray.job_config import JobConfig
from ray.util.annotations import Deprecated
from ray.util.client import ray
from ray._private.utils import get_ray_doc_version
logger = logging.getLogger(__name__)

@Deprecated(message='Use ray.init(ray://<head_node_ip_address>:<ray_client_server_port>) instead. See detailed usage at {}.'.format(f'https://docs.ray.io/en/{get_ray_doc_version()}/ray-core/package-ref.html#ray-init'))
def connect(conn_str: str, secure: bool=False, metadata: List[Tuple[str, str]]=None, connection_retries: int=3, job_config: JobConfig=None, namespace: str=None, *, ignore_version: bool=False, _credentials: Optional['grpc.ChannelCredentials']=None, ray_init_kwargs: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    if ray.is_connected():
        ignore_reinit_error = ray_init_kwargs.get('ignore_reinit_error', False)
        if ignore_reinit_error:
            logger.info('Calling ray.init() again after it has already been called. Reusing the existing Ray client connection.')
            return ray.get_context().client_worker.connection_info()
        raise RuntimeError('Ray Client is already connected. Maybe you called ray.init("ray://<address>") twice by accident?')
    _set_client_hook_status(True)
    _explicitly_enable_client_mode()
    conn = ray.connect(conn_str, job_config=job_config, secure=secure, metadata=metadata, connection_retries=connection_retries, namespace=namespace, ignore_version=ignore_version, _credentials=_credentials, ray_init_kwargs=ray_init_kwargs)
    return conn

@Deprecated(message='Use ray.shutdown() instead. See detailed usage at {}.'.format(f'https://docs.ray.io/en/{get_ray_doc_version()}/ray-core/package-ref.html#ray-shutdown'))
def disconnect():
    if False:
        while True:
            i = 10
    'Disconnects from server; is idempotent.'
    return ray.disconnect()