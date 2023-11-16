from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
from deeplake.core.vectorstore.deepmemory_vectorstore import DeepMemoryVectorStore
from deeplake.client.client import DeepMemoryBackendClient
from deeplake.util.path import get_path_type

def vectorstore_factory(path, *args, **kwargs):
    if False:
        while True:
            i = 10
    path_type = get_path_type(path)
    dm_client = DeepMemoryBackendClient(token=kwargs.get('token'))
    user_profile = dm_client.get_user_profile()
    if path_type == 'hub':
        dataset_id = path[6:].split('/')[0]
    else:
        dataset_id = user_profile['name']
    deepmemory_is_available = dm_client.deepmemory_is_available(dataset_id)
    if deepmemory_is_available:
        return DeepMemoryVectorStore(*args, path=path, client=dm_client, org_id=dataset_id, **kwargs)
    return VectorStore(*args, path=path, **kwargs)