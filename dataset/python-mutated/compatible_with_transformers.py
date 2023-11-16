import transformers
from packaging import version

def compatible_position_ids(state_dict, position_id_key):
    if False:
        i = 10
        return i + 15
    'Transformers no longer expect position_ids after transformers==4.31\n       https://github.com/huggingface/transformers/pull/24505\n\n    Args:\n        position_id_key (str): position_ids key,\n            such as(encoder.embeddings.position_ids)\n    '
    transformer_version = version.parse('.'.join(transformers.__version__.split('.')[:2]))
    if transformer_version >= version.parse('4.31.0'):
        del state_dict[position_id_key]