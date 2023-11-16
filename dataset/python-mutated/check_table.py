"""
Utility that checks the big table in the file docs/source/en/index.md and potentially updates it.

Use from the root of the repo with:

```bash
python utils/check_inits.py
```

for a check that will error in case of inconsistencies (used by `make repo-consistency`).

To auto-fix issues run:

```bash
python utils/check_inits.py --fix_and_overwrite
```

which is used by `make fix-copies`.
"""
import argparse
import collections
import os
import re
from typing import List
from transformers.utils import direct_transformers_import
TRANSFORMERS_PATH = 'src/transformers'
PATH_TO_DOCS = 'docs/source/en'
REPO_PATH = '.'

def _find_text_in_file(filename: str, start_prompt: str, end_prompt: str) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Find the text in filename between two prompts.\n\n    Args:\n        filename (`str`): The file to search into.\n        start_prompt (`str`): A string to look for at the start of the content searched.\n        end_prompt (`str`): A string that will mark the end of the content to look for.\n\n    Returns:\n        `str`: The content between the prompts.\n    '
    with open(filename, 'r', encoding='utf-8', newline='\n') as f:
        lines = f.readlines()
    start_index = 0
    while not lines[start_index].startswith(start_prompt):
        start_index += 1
    start_index += 1
    end_index = start_index
    while not lines[end_index].startswith(end_prompt):
        end_index += 1
    end_index -= 1
    while len(lines[start_index]) <= 1:
        start_index += 1
    while len(lines[end_index]) <= 1:
        end_index -= 1
    end_index += 1
    return (''.join(lines[start_index:end_index]), start_index, end_index, lines)
_re_tf_models = re.compile('TF(.*)(?:Model|Encoder|Decoder|ForConditionalGeneration)')
_re_flax_models = re.compile('Flax(.*)(?:Model|Encoder|Decoder|ForConditionalGeneration)')
_re_pt_models = re.compile('(.*)(?:Model|Encoder|Decoder|ForConditionalGeneration)')
transformers_module = direct_transformers_import(TRANSFORMERS_PATH)

def camel_case_split(identifier: str) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Split a camel-cased name into words.\n\n    Args:\n        identifier (`str`): The camel-cased name to parse.\n\n    Returns:\n        `List[str]`: The list of words in the identifier (as seprated by capital letters).\n\n    Example:\n\n    ```py\n    >>> camel_case_split("CamelCasedClass")\n    ["Camel", "Cased", "Class"]\n    ```\n    '
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def _center_text(text: str, width: int) -> str:
    if False:
        print('Hello World!')
    '\n    Utility that will add spaces on the left and right of a text to make it centered for a given width.\n\n    Args:\n        text (`str`): The text to center.\n        width (`int`): The desired length of the result.\n\n    Returns:\n        `str`: A text of length `width` with the original `text` in the middle.\n    '
    text_length = 2 if text == '✅' or text == '❌' else len(text)
    left_indent = (width - text_length) // 2
    right_indent = width - text_length - left_indent
    return ' ' * left_indent + text + ' ' * right_indent
SPECIAL_MODEL_NAME_LINK_MAPPING = {'Data2VecAudio': '[Data2VecAudio](model_doc/data2vec)', 'Data2VecText': '[Data2VecText](model_doc/data2vec)', 'Data2VecVision': '[Data2VecVision](model_doc/data2vec)', 'DonutSwin': '[DonutSwin](model_doc/donut)'}
MODEL_NAMES_WITH_SAME_CONFIG = {'BARThez': 'BART', 'BARTpho': 'BART', 'BertJapanese': 'BERT', 'BERTweet': 'BERT', 'BORT': 'BERT', 'ByT5': 'T5', 'CPM': 'OpenAI GPT-2', 'DePlot': 'Pix2Struct', 'DialoGPT': 'OpenAI GPT-2', 'DiT': 'BEiT', 'FLAN-T5': 'T5', 'FLAN-UL2': 'T5', 'HerBERT': 'BERT', 'LayoutXLM': 'LayoutLMv2', 'Llama2': 'LLaMA', 'MatCha': 'Pix2Struct', 'mBART-50': 'mBART', 'Megatron-GPT2': 'OpenAI GPT-2', 'mLUKE': 'LUKE', 'MMS': 'Wav2Vec2', 'NLLB': 'M2M100', 'PhoBERT': 'BERT', 'T5v1.1': 'T5', 'TAPEX': 'BART', 'UL2': 'T5', 'Wav2Vec2Phoneme': 'Wav2Vec2', 'XLM-V': 'XLM-RoBERTa', 'XLS-R': 'Wav2Vec2', 'XLSR-Wav2Vec2': 'Wav2Vec2'}

def get_model_table_from_auto_modules() -> str:
    if False:
        i = 10
        return i + 15
    '\n    Generates an up-to-date model table from the content of the auto modules.\n    '
    config_maping_names = transformers_module.models.auto.configuration_auto.CONFIG_MAPPING_NAMES
    model_name_to_config = {name: config_maping_names[code] for (code, name) in transformers_module.MODEL_NAMES_MAPPING.items() if code in config_maping_names}
    model_name_to_prefix = {name: config.replace('Config', '') for (name, config) in model_name_to_config.items()}
    pt_models = collections.defaultdict(bool)
    tf_models = collections.defaultdict(bool)
    flax_models = collections.defaultdict(bool)
    for attr_name in dir(transformers_module):
        lookup_dict = None
        if _re_tf_models.match(attr_name) is not None:
            lookup_dict = tf_models
            attr_name = _re_tf_models.match(attr_name).groups()[0]
        elif _re_flax_models.match(attr_name) is not None:
            lookup_dict = flax_models
            attr_name = _re_flax_models.match(attr_name).groups()[0]
        elif _re_pt_models.match(attr_name) is not None:
            lookup_dict = pt_models
            attr_name = _re_pt_models.match(attr_name).groups()[0]
        if lookup_dict is not None:
            while len(attr_name) > 0:
                if attr_name in model_name_to_prefix.values():
                    lookup_dict[attr_name] = True
                    break
                attr_name = ''.join(camel_case_split(attr_name)[:-1])
    model_names = list(model_name_to_config.keys()) + list(MODEL_NAMES_WITH_SAME_CONFIG.keys())
    model_names_mapping = transformers_module.models.auto.configuration_auto.MODEL_NAMES_MAPPING
    model_name_to_link_mapping = {value: f'[{value}](model_doc/{key})' for (key, value) in model_names_mapping.items()}
    model_name_to_link_mapping = {k: SPECIAL_MODEL_NAME_LINK_MAPPING[k] if k in SPECIAL_MODEL_NAME_LINK_MAPPING else v for (k, v) in model_name_to_link_mapping.items()}
    names_to_exclude = ['MaskFormerSwin', 'TimmBackbone', 'Speech2Text2']
    model_names = [name for name in model_names if name not in names_to_exclude]
    model_names.sort(key=str.lower)
    columns = ['Model', 'PyTorch support', 'TensorFlow support', 'Flax Support']
    widths = [len(c) + 2 for c in columns]
    widths[0] = max([len(doc_link) for doc_link in model_name_to_link_mapping.values()]) + 2
    table = '|' + '|'.join([_center_text(c, w) for (c, w) in zip(columns, widths)]) + '|\n'
    table += '|' + '|'.join([':' + '-' * (w - 2) + ':' for w in widths]) + '|\n'
    check = {True: '✅', False: '❌'}
    for name in model_names:
        if name in MODEL_NAMES_WITH_SAME_CONFIG.keys():
            prefix = model_name_to_prefix[MODEL_NAMES_WITH_SAME_CONFIG[name]]
        else:
            prefix = model_name_to_prefix[name]
        line = [model_name_to_link_mapping[name], check[pt_models[prefix]], check[tf_models[prefix]], check[flax_models[prefix]]]
        table += '|' + '|'.join([_center_text(l, w) for (l, w) in zip(line, widths)]) + '|\n'
    return table

def check_model_table(overwrite=False):
    if False:
        i = 10
        return i + 15
    "\n    Check the model table in the index.md is consistent with the state of the lib and potentially fix it.\n\n    Args:\n        overwrite (`bool`, *optional*, defaults to `False`):\n            Whether or not to overwrite the table when it's not up to date.\n    "
    (current_table, start_index, end_index, lines) = _find_text_in_file(filename=os.path.join(PATH_TO_DOCS, 'index.md'), start_prompt='<!--This table is updated automatically from the auto modules', end_prompt='<!-- End table-->')
    new_table = get_model_table_from_auto_modules()
    if current_table != new_table:
        if overwrite:
            with open(os.path.join(PATH_TO_DOCS, 'index.md'), 'w', encoding='utf-8', newline='\n') as f:
                f.writelines(lines[:start_index] + [new_table] + lines[end_index:])
        else:
            raise ValueError('The model table in the `index.md` has not been updated. Run `make fix-copies` to fix this.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix_and_overwrite', action='store_true', help='Whether to fix inconsistencies.')
    args = parser.parse_args()
    check_model_table(args.fix_and_overwrite)