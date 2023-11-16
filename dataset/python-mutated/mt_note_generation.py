"""
MT Note Generation is a set of synthetic dialogues between Assistant and
User where the user asks the assistant to generate a clinical note for a patient persona.
"""
import json
from typing import Dict, List, Tuple
import datasets
from .hub import OpenAssistantConfig, features
_CITATION = ' @misc{transcribed medical transcription sample reports and examples, title={Welcome to MTSamples},\n url={https://mtsamples.com/},\n journal={Transcribed Medical Transcription Sample Reports and Examples}}\n'
_DATASETNAME = 'mt_note_generation'
_DISPLAYNAME = 'MT Samples Note Generation'
_DESCRIPTION = 'A dataset of instructions for generating clinical notes from MT samples.\n'
_HOMEPAGE = ''
_LICENSE = 'mit'
_URLS = {_DATASETNAME: {'train': './data/mt_note_generation_train.jsonl', 'test': './data/mt_note_generation_test.jsonl', 'validation': './data/mt_note_generation_validation.jsonl'}}
_SUPPORTED_TASKS = ['dialogue-modeling']
_VERSION = '1.0.0'

class MTNoteGenerationDataset(datasets.GeneratorBasedBuilder):
    """A set of dialogues synthesized from the MT Samples dataset."""
    VERSION = datasets.Version(_VERSION)
    BUILDER_CONFIGS = [OpenAssistantConfig(name=f'{_DATASETNAME}_dialogue_modeling', version=VERSION, description=f'OpenAssistant dataset config for {_DATASETNAME}', schema='dialogue_modeling', subset_id=_DATASETNAME)]
    DEFAULT_CONFIG_NAME = f'{_DATASETNAME}_dialogue_modeling'

    def _info(self) -> datasets.DatasetInfo:
        if False:
            while True:
                i = 10
        return datasets.DatasetInfo(description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION)

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        if False:
            return 10
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'filepath': data_dir, 'split': 'train'}), datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={'filepath': data_dir, 'split': 'test'}), datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={'filepath': data_dir, 'split': 'validation'})]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        if False:
            for i in range(10):
                print('nop')
        'Yields examples as (key, example) tuples.'
        if self.config.schema == 'dialogue_modeling':
            key = 0
            with open(filepath[split], 'r', encoding='utf8') as data:
                while True:
                    line = data.readline()
                    if not line:
                        return
                    yield (key, json.loads(line))
                    key += 1