from typing import Dict as TypeDict
from typing import List as TypeList
from typing import Optional
from .grammar import GrammarTerm
from .grammar import GrammarVerb
from .grammar import HostGrammarTerm

def get_land_verb() -> GrammarVerb:
    if False:
        for i in range(10):
            print('nop')
    full_sentence = [{'name': 'node_name', 'type': 'adjective', 'klass': GrammarTerm, 'example': "'my_domain'"}, {'name': 'preposition', 'type': 'preposition', 'klass': GrammarTerm, 'default': 'at', 'options': ['at', 'on']}, {'name': 'host', 'type': 'propernoun', 'klass': HostGrammarTerm, 'default': 'docker', 'example': 'docker'}]
    abbreviations: TypeDict[int, TypeList[Optional[str]]] = {3: ['adjective', 'preposition', 'propernoun'], 2: ['adjective', None, 'propernoun'], 1: ['adjective', None, None], 0: [None, None, None]}
    return GrammarVerb(command='land', full_sentence=full_sentence, abbreviations=abbreviations)