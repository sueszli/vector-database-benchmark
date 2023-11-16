import logging
import json
import os
from packaging import version
import re
from typing import Any, Dict, Union
import ray
from ray.rllib.utils.serialization import NOT_SERIALIZABLE, serialize_type
from ray.train import Checkpoint
from ray.util import log_once
from ray.util.annotations import PublicAPI
logger = logging.getLogger(__name__)
CHECKPOINT_VERSION = version.Version('1.1')
CHECKPOINT_VERSION_LEARNER = version.Version('1.2')

@PublicAPI(stability='alpha')
def get_checkpoint_info(checkpoint: Union[str, Checkpoint]) -> Dict[str, Any]:
    if False:
        while True:
            i = 10
    'Returns a dict with information about a Algorithm/Policy checkpoint.\n\n    If the given checkpoint is a >=v1.0 checkpoint directory, try reading all\n    information from the contained `rllib_checkpoint.json` file.\n\n    Args:\n        checkpoint: The checkpoint directory (str) or an AIR Checkpoint object.\n\n    Returns:\n        A dict containing the keys:\n        "type": One of "Policy" or "Algorithm".\n        "checkpoint_version": A version tuple, e.g. v1.0, indicating the checkpoint\n        version. This will help RLlib to remain backward compatible wrt. future\n        Ray and checkpoint versions.\n        "checkpoint_dir": The directory with all the checkpoint files in it. This might\n        be the same as the incoming `checkpoint` arg.\n        "state_file": The main file with the Algorithm/Policy\'s state information in it.\n        This is usually a pickle-encoded file.\n        "policy_ids": An optional set of PolicyIDs in case we are dealing with an\n        Algorithm checkpoint. None if `checkpoint` is a Policy checkpoint.\n    '
    info = {'type': 'Algorithm', 'format': 'cloudpickle', 'checkpoint_version': CHECKPOINT_VERSION, 'checkpoint_dir': None, 'state_file': None, 'policy_ids': None}
    if isinstance(checkpoint, Checkpoint):
        checkpoint: str = checkpoint.to_directory()
    if os.path.isdir(checkpoint):
        info.update({'checkpoint_dir': checkpoint})
        for file in os.listdir(checkpoint):
            path_file = os.path.join(checkpoint, file)
            if os.path.isfile(path_file):
                if re.match('checkpoint-\\d+', file):
                    info.update({'checkpoint_version': version.Version('0.1'), 'state_file': path_file})
                    return info
        if os.path.isfile(os.path.join(checkpoint, 'rllib_checkpoint.json')):
            with open(os.path.join(checkpoint, 'rllib_checkpoint.json')) as f:
                rllib_checkpoint_info = json.load(fp=f)
            if 'checkpoint_version' in rllib_checkpoint_info:
                rllib_checkpoint_info['checkpoint_version'] = version.Version(rllib_checkpoint_info['checkpoint_version'])
            info.update(rllib_checkpoint_info)
        elif log_once('no_rllib_checkpoint_json_file'):
            logger.warning(f'No `rllib_checkpoint.json` file found in checkpoint directory {checkpoint}! Trying to extract checkpoint info from other files found in that dir.')
        for extension in ['pkl', 'msgpck']:
            if os.path.isfile(os.path.join(checkpoint, 'policy_state.' + extension)):
                info.update({'type': 'Policy', 'format': 'cloudpickle' if extension == 'pkl' else 'msgpack', 'checkpoint_version': CHECKPOINT_VERSION, 'state_file': os.path.join(checkpoint, f'policy_state.{extension}')})
                return info
        format = None
        for extension in ['pkl', 'msgpck']:
            state_file = os.path.join(checkpoint, f'algorithm_state.{extension}')
            if os.path.isfile(state_file):
                format = 'cloudpickle' if extension == 'pkl' else 'msgpack'
                break
        if format is None:
            raise ValueError('Given checkpoint does not seem to be valid! No file with the name `algorithm_state.[pkl|msgpck]` (or `checkpoint-[0-9]+`) found.')
        info.update({'format': format, 'state_file': state_file})
        policies_dir = os.path.join(checkpoint, 'policies')
        if os.path.isdir(policies_dir):
            policy_ids = set()
            for policy_id in os.listdir(policies_dir):
                policy_ids.add(policy_id)
            info.update({'policy_ids': policy_ids})
    elif os.path.isfile(checkpoint):
        info.update({'checkpoint_version': version.Version('0.1'), 'checkpoint_dir': os.path.dirname(checkpoint), 'state_file': checkpoint})
    else:
        raise ValueError(f'Given checkpoint ({checkpoint}) not found! Must be a checkpoint directory (or a file for older checkpoint versions).')
    return info

@PublicAPI(stability='beta')
def convert_to_msgpack_checkpoint(checkpoint: Union[str, Checkpoint], msgpack_checkpoint_dir: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Converts an Algorithm checkpoint (pickle based) to a msgpack based one.\n\n    Msgpack has the advantage of being python version independent.\n\n    Args:\n        checkpoint: The directory, in which to find the Algorithm checkpoint (pickle\n            based).\n        msgpack_checkpoint_dir: The directory, in which to create the new msgpack\n            based checkpoint.\n\n    Returns:\n        The directory in which the msgpack checkpoint has been created. Note that\n        this is the same as `msgpack_checkpoint_dir`.\n    '
    from ray.rllib.algorithms import Algorithm
    from ray.rllib.utils.policy import validate_policy_id
    msgpack = try_import_msgpack(error=True)
    algo = Algorithm.from_checkpoint(checkpoint)
    state = algo.__getstate__()
    state['algorithm_class'] = serialize_type(state['algorithm_class'])
    state['config'] = state['config'].serialize()
    policy_states = {}
    if 'worker' in state and 'policy_states' in state['worker']:
        policy_states = state['worker'].pop('policy_states', {})
    state['worker']['policy_mapping_fn'] = NOT_SERIALIZABLE
    state['worker']['is_policy_to_train'] = NOT_SERIALIZABLE
    if state['config']['_enable_new_api_stack']:
        state['checkpoint_version'] = str(CHECKPOINT_VERSION_LEARNER)
    else:
        state['checkpoint_version'] = str(CHECKPOINT_VERSION)
    state_file = os.path.join(msgpack_checkpoint_dir, 'algorithm_state.msgpck')
    with open(state_file, 'wb') as f:
        msgpack.dump(state, f)
    with open(os.path.join(msgpack_checkpoint_dir, 'rllib_checkpoint.json'), 'w') as f:
        json.dump({'type': 'Algorithm', 'checkpoint_version': state['checkpoint_version'], 'format': 'msgpack', 'state_file': state_file, 'policy_ids': list(policy_states.keys()), 'ray_version': ray.__version__, 'ray_commit': ray.__commit__}, f)
    for (pid, policy_state) in policy_states.items():
        validate_policy_id(pid, error=True)
        policy_dir = os.path.join(msgpack_checkpoint_dir, 'policies', pid)
        os.makedirs(policy_dir, exist_ok=True)
        policy = algo.get_policy(pid)
        policy.export_checkpoint(policy_dir, policy_state=policy_state, checkpoint_format='msgpack')
    algo.stop()
    return msgpack_checkpoint_dir

@PublicAPI(stability='beta')
def convert_to_msgpack_policy_checkpoint(policy_checkpoint: Union[str, Checkpoint], msgpack_checkpoint_dir: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Converts a Policy checkpoint (pickle based) to a msgpack based one.\n\n    Msgpack has the advantage of being python version independent.\n\n    Args:\n        policy_checkpoint: The directory, in which to find the Policy checkpoint (pickle\n            based).\n        msgpack_checkpoint_dir: The directory, in which to create the new msgpack\n            based checkpoint.\n\n    Returns:\n        The directory in which the msgpack checkpoint has been created. Note that\n        this is the same as `msgpack_checkpoint_dir`.\n    '
    from ray.rllib.policy.policy import Policy
    policy = Policy.from_checkpoint(policy_checkpoint)
    os.makedirs(msgpack_checkpoint_dir, exist_ok=True)
    policy.export_checkpoint(msgpack_checkpoint_dir, policy_state=policy.get_state(), checkpoint_format='msgpack')
    del policy
    return msgpack_checkpoint_dir

@PublicAPI
def try_import_msgpack(error: bool=False):
    if False:
        while True:
            i = 10
    'Tries importing msgpack and msgpack_numpy and returns the patched msgpack module.\n\n    Returns None if error is False and msgpack or msgpack_numpy is not installed.\n    Raises an error, if error is True and the modules could not be imported.\n\n    Args:\n        error: Whether to raise an error if msgpack/msgpack_numpy cannot be imported.\n\n    Returns:\n        The `msgpack` module.\n\n    Raises:\n        ImportError: If error=True and msgpack/msgpack_numpy is not installed.\n    '
    try:
        import msgpack
        import msgpack_numpy
        msgpack_numpy.patch()
        return msgpack
    except Exception:
        if error:
            raise ImportError('Could not import or setup msgpack and msgpack_numpy! Try running `pip install msgpack msgpack_numpy` first.')