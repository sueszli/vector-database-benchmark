import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import srsly
from ..errors import Errors, Warnings
from ..language import Language
from ..matcher import Matcher, PhraseMatcher
from ..matcher.levenshtein import levenshtein_compare
from ..scorer import get_ner_prf
from ..tokens import Doc, Span
from ..training import Example
from ..util import SimpleFrozenList, ensure_path, from_disk, registry, to_disk
from .pipe import Pipe
DEFAULT_ENT_ID_SEP = '||'
PatternType = Dict[str, Union[str, List[Dict[str, Any]]]]

@Language.factory('entity_ruler', assigns=['doc.ents', 'token.ent_type', 'token.ent_iob'], default_config={'phrase_matcher_attr': None, 'matcher_fuzzy_compare': {'@misc': 'spacy.levenshtein_compare.v1'}, 'validate': False, 'overwrite_ents': False, 'ent_id_sep': DEFAULT_ENT_ID_SEP, 'scorer': {'@scorers': 'spacy.entity_ruler_scorer.v1'}}, default_score_weights={'ents_f': 1.0, 'ents_p': 0.0, 'ents_r': 0.0, 'ents_per_type': None})
def make_entity_ruler(nlp: Language, name: str, phrase_matcher_attr: Optional[Union[int, str]], matcher_fuzzy_compare: Callable, validate: bool, overwrite_ents: bool, ent_id_sep: str, scorer: Optional[Callable]):
    if False:
        print('Hello World!')
    return EntityRuler(nlp, name, phrase_matcher_attr=phrase_matcher_attr, matcher_fuzzy_compare=matcher_fuzzy_compare, validate=validate, overwrite_ents=overwrite_ents, ent_id_sep=ent_id_sep, scorer=scorer)

def entity_ruler_score(examples, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return get_ner_prf(examples)

@registry.scorers('spacy.entity_ruler_scorer.v1')
def make_entity_ruler_scorer():
    if False:
        i = 10
        return i + 15
    return entity_ruler_score

class EntityRuler(Pipe):
    """The EntityRuler lets you add spans to the `Doc.ents` using token-based
    rules or exact phrase matches. It can be combined with the statistical
    `EntityRecognizer` to boost accuracy, or used on its own to implement a
    purely rule-based entity recognition system. After initialization, the
    component is typically added to the pipeline using `nlp.add_pipe`.

    DOCS: https://spacy.io/api/entityruler
    USAGE: https://spacy.io/usage/rule-based-matching#entityruler
    """

    def __init__(self, nlp: Language, name: str='entity_ruler', *, phrase_matcher_attr: Optional[Union[int, str]]=None, matcher_fuzzy_compare: Callable=levenshtein_compare, validate: bool=False, overwrite_ents: bool=False, ent_id_sep: str=DEFAULT_ENT_ID_SEP, patterns: Optional[List[PatternType]]=None, scorer: Optional[Callable]=entity_ruler_score) -> None:
        if False:
            print('Hello World!')
        'Initialize the entity ruler. If patterns are supplied here, they\n        need to be a list of dictionaries with a `"label"` and `"pattern"`\n        key. A pattern can either be a token pattern (list) or a phrase pattern\n        (string). For example: `{\'label\': \'ORG\', \'pattern\': \'Apple\'}`.\n\n        nlp (Language): The shared nlp object to pass the vocab to the matchers\n            and process phrase patterns.\n        name (str): Instance name of the current pipeline component. Typically\n            passed in automatically from the factory when the component is\n            added. Used to disable the current entity ruler while creating\n            phrase patterns with the nlp object.\n        phrase_matcher_attr (int / str): Token attribute to match on, passed\n            to the internal PhraseMatcher as `attr`.\n        matcher_fuzzy_compare (Callable): The fuzzy comparison method for the\n            internal Matcher. Defaults to\n            spacy.matcher.levenshtein.levenshtein_compare.\n        validate (bool): Whether patterns should be validated, passed to\n            Matcher and PhraseMatcher as `validate`\n        patterns (iterable): Optional patterns to load in.\n        overwrite_ents (bool): If existing entities are present, e.g. entities\n            added by the model, overwrite them by matches if necessary.\n        ent_id_sep (str): Separator used internally for entity IDs.\n        scorer (Optional[Callable]): The scoring method. Defaults to\n            spacy.scorer.get_ner_prf.\n\n        DOCS: https://spacy.io/api/entityruler#init\n        '
        self.nlp = nlp
        self.name = name
        self.overwrite = overwrite_ents
        self.token_patterns = defaultdict(list)
        self.phrase_patterns = defaultdict(list)
        self._validate = validate
        self.matcher_fuzzy_compare = matcher_fuzzy_compare
        self.matcher = Matcher(nlp.vocab, validate=validate, fuzzy_compare=self.matcher_fuzzy_compare)
        self.phrase_matcher_attr = phrase_matcher_attr
        self.phrase_matcher = PhraseMatcher(nlp.vocab, attr=self.phrase_matcher_attr, validate=validate)
        self.ent_id_sep = ent_id_sep
        self._ent_ids = defaultdict(tuple)
        if patterns is not None:
            self.add_patterns(patterns)
        self.scorer = scorer

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        'The number of all patterns added to the entity ruler.'
        n_token_patterns = sum((len(p) for p in self.token_patterns.values()))
        n_phrase_patterns = sum((len(p) for p in self.phrase_patterns.values()))
        return n_token_patterns + n_phrase_patterns

    def __contains__(self, label: str) -> bool:
        if False:
            return 10
        'Whether a label is present in the patterns.'
        return label in self.token_patterns or label in self.phrase_patterns

    def __call__(self, doc: Doc) -> Doc:
        if False:
            for i in range(10):
                print('nop')
        'Find matches in document and add them as entities.\n\n        doc (Doc): The Doc object in the pipeline.\n        RETURNS (Doc): The Doc with added entities, if available.\n\n        DOCS: https://spacy.io/api/entityruler#call\n        '
        error_handler = self.get_error_handler()
        try:
            matches = self.match(doc)
            self.set_annotations(doc, matches)
            return doc
        except Exception as e:
            return error_handler(self.name, self, [doc], e)

    def match(self, doc: Doc):
        if False:
            print('Hello World!')
        self._require_patterns()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='\\[W036')
            matches = list(self.matcher(doc)) + list(self.phrase_matcher(doc))
        final_matches = set([(m_id, start, end) for (m_id, start, end) in matches if start != end])
        get_sort_key = lambda m: (m[2] - m[1], -m[1])
        final_matches = sorted(final_matches, key=get_sort_key, reverse=True)
        return final_matches

    def set_annotations(self, doc, matches):
        if False:
            print('Hello World!')
        'Modify the document in place'
        entities = list(doc.ents)
        new_entities = []
        seen_tokens = set()
        for (match_id, start, end) in matches:
            if any((t.ent_type for t in doc[start:end])) and (not self.overwrite):
                continue
            if start not in seen_tokens and end - 1 not in seen_tokens:
                if match_id in self._ent_ids:
                    (label, ent_id) = self._ent_ids[match_id]
                    span = Span(doc, start, end, label=label, span_id=ent_id)
                else:
                    span = Span(doc, start, end, label=match_id)
                new_entities.append(span)
                entities = [e for e in entities if not (e.start < end and e.end > start)]
                seen_tokens.update(range(start, end))
        doc.ents = entities + new_entities

    @property
    def labels(self) -> Tuple[str, ...]:
        if False:
            for i in range(10):
                print('nop')
        'All labels present in the match patterns.\n\n        RETURNS (set): The string labels.\n\n        DOCS: https://spacy.io/api/entityruler#labels\n        '
        keys = set(self.token_patterns.keys())
        keys.update(self.phrase_patterns.keys())
        all_labels = set()
        for l in keys:
            if self.ent_id_sep in l:
                (label, _) = self._split_label(l)
                all_labels.add(label)
            else:
                all_labels.add(l)
        return tuple(sorted(all_labels))

    def initialize(self, get_examples: Callable[[], Iterable[Example]], *, nlp: Optional[Language]=None, patterns: Optional[Sequence[PatternType]]=None):
        if False:
            print('Hello World!')
        'Initialize the pipe for training.\n\n        get_examples (Callable[[], Iterable[Example]]): Function that\n            returns a representative sample of gold-standard Example objects.\n        nlp (Language): The current nlp object the component is part of.\n        patterns Optional[Iterable[PatternType]]: The list of patterns.\n\n        DOCS: https://spacy.io/api/entityruler#initialize\n        '
        self.clear()
        if patterns:
            self.add_patterns(patterns)

    @property
    def ent_ids(self) -> Tuple[Optional[str], ...]:
        if False:
            i = 10
            return i + 15
        'All entity ids present in the match patterns `id` properties\n\n        RETURNS (set): The string entity ids.\n\n        DOCS: https://spacy.io/api/entityruler#ent_ids\n        '
        keys = set(self.token_patterns.keys())
        keys.update(self.phrase_patterns.keys())
        all_ent_ids = set()
        for l in keys:
            if self.ent_id_sep in l:
                (_, ent_id) = self._split_label(l)
                all_ent_ids.add(ent_id)
        return tuple(all_ent_ids)

    @property
    def patterns(self) -> List[PatternType]:
        if False:
            while True:
                i = 10
        'Get all patterns that were added to the entity ruler.\n\n        RETURNS (list): The original patterns, one dictionary per pattern.\n\n        DOCS: https://spacy.io/api/entityruler#patterns\n        '
        all_patterns = []
        for (label, patterns) in self.token_patterns.items():
            for pattern in patterns:
                (ent_label, ent_id) = self._split_label(label)
                p = {'label': ent_label, 'pattern': pattern}
                if ent_id:
                    p['id'] = ent_id
                all_patterns.append(p)
        for (label, patterns) in self.phrase_patterns.items():
            for pattern in patterns:
                (ent_label, ent_id) = self._split_label(label)
                p = {'label': ent_label, 'pattern': pattern.text}
                if ent_id:
                    p['id'] = ent_id
                all_patterns.append(p)
        return all_patterns

    def add_patterns(self, patterns: List[PatternType]) -> None:
        if False:
            i = 10
            return i + 15
        "Add patterns to the entity ruler. A pattern can either be a token\n        pattern (list of dicts) or a phrase pattern (string). For example:\n        {'label': 'ORG', 'pattern': 'Apple'}\n        {'label': 'GPE', 'pattern': [{'lower': 'san'}, {'lower': 'francisco'}]}\n\n        patterns (list): The patterns to add.\n\n        DOCS: https://spacy.io/api/entityruler#add_patterns\n        "
        try:
            current_index = -1
            for (i, (name, pipe)) in enumerate(self.nlp.pipeline):
                if self == pipe:
                    current_index = i
                    break
            subsequent_pipes = [pipe for pipe in self.nlp.pipe_names[current_index:]]
        except ValueError:
            subsequent_pipes = []
        with self.nlp.select_pipes(disable=subsequent_pipes):
            token_patterns = []
            phrase_pattern_labels = []
            phrase_pattern_texts = []
            phrase_pattern_ids = []
            for entry in patterns:
                if isinstance(entry['pattern'], str):
                    phrase_pattern_labels.append(entry['label'])
                    phrase_pattern_texts.append(entry['pattern'])
                    phrase_pattern_ids.append(entry.get('id'))
                elif isinstance(entry['pattern'], list):
                    token_patterns.append(entry)
            phrase_patterns = []
            for (label, pattern, ent_id) in zip(phrase_pattern_labels, self.nlp.pipe(phrase_pattern_texts), phrase_pattern_ids):
                phrase_pattern = {'label': label, 'pattern': pattern}
                if ent_id:
                    phrase_pattern['id'] = ent_id
                phrase_patterns.append(phrase_pattern)
            for entry in token_patterns + phrase_patterns:
                label = entry['label']
                if 'id' in entry:
                    ent_label = label
                    label = self._create_label(label, entry['id'])
                    key = self.matcher._normalize_key(label)
                    self._ent_ids[key] = (ent_label, entry['id'])
                pattern = entry['pattern']
                if isinstance(pattern, Doc):
                    self.phrase_patterns[label].append(pattern)
                    self.phrase_matcher.add(label, [pattern])
                elif isinstance(pattern, list):
                    self.token_patterns[label].append(pattern)
                    self.matcher.add(label, [pattern])
                else:
                    raise ValueError(Errors.E097.format(pattern=pattern))

    def clear(self) -> None:
        if False:
            i = 10
            return i + 15
        'Reset all patterns.'
        self.token_patterns = defaultdict(list)
        self.phrase_patterns = defaultdict(list)
        self._ent_ids = defaultdict(tuple)
        self.matcher = Matcher(self.nlp.vocab, validate=self._validate, fuzzy_compare=self.matcher_fuzzy_compare)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr=self.phrase_matcher_attr, validate=self._validate)

    def remove(self, ent_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Remove a pattern by its ent_id if a pattern with this ent_id was added before\n\n        ent_id (str): id of the pattern to be removed\n        RETURNS: None\n        DOCS: https://spacy.io/api/entityruler#remove\n        '
        label_id_pairs = [(label, eid) for (label, eid) in self._ent_ids.values() if eid == ent_id]
        if not label_id_pairs:
            raise ValueError(Errors.E1024.format(attr_type='ID', label=ent_id, component=self.name))
        created_labels = [self._create_label(label, eid) for (label, eid) in label_id_pairs]
        self.phrase_patterns = defaultdict(list, {label: val for (label, val) in self.phrase_patterns.items() if label not in created_labels})
        self.token_patterns = defaultdict(list, {label: val for (label, val) in self.token_patterns.items() if label not in created_labels})
        for label in created_labels:
            if label in self.phrase_matcher:
                self.phrase_matcher.remove(label)
            else:
                self.matcher.remove(label)

    def _require_patterns(self) -> None:
        if False:
            return 10
        'Raise a warning if this component has no patterns defined.'
        if len(self) == 0:
            warnings.warn(Warnings.W036.format(name=self.name))

    def _split_label(self, label: str) -> Tuple[str, Optional[str]]:
        if False:
            for i in range(10):
                print('nop')
        'Split Entity label into ent_label and ent_id if it contains self.ent_id_sep\n\n        label (str): The value of label in a pattern entry\n        RETURNS (tuple): ent_label, ent_id\n        '
        if self.ent_id_sep in label:
            (ent_label, ent_id) = label.rsplit(self.ent_id_sep, 1)
        else:
            ent_label = label
            ent_id = None
        return (ent_label, ent_id)

    def _create_label(self, label: Any, ent_id: Any) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Join Entity label with ent_id if the pattern has an `id` attribute\n        If ent_id is not a string, the label is returned as is.\n\n        label (str): The label to set for ent.label_\n        ent_id (str): The label\n        RETURNS (str): The ent_label joined with configured `ent_id_sep`\n        '
        if isinstance(ent_id, str):
            label = f'{label}{self.ent_id_sep}{ent_id}'
        return label

    def from_bytes(self, patterns_bytes: bytes, *, exclude: Iterable[str]=SimpleFrozenList()) -> 'EntityRuler':
        if False:
            while True:
                i = 10
        'Load the entity ruler from a bytestring.\n\n        patterns_bytes (bytes): The bytestring to load.\n        RETURNS (EntityRuler): The loaded entity ruler.\n\n        DOCS: https://spacy.io/api/entityruler#from_bytes\n        '
        cfg = srsly.msgpack_loads(patterns_bytes)
        self.clear()
        if isinstance(cfg, dict):
            self.add_patterns(cfg.get('patterns', cfg))
            self.overwrite = cfg.get('overwrite', False)
            self.phrase_matcher_attr = cfg.get('phrase_matcher_attr', None)
            self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr=self.phrase_matcher_attr)
            self.ent_id_sep = cfg.get('ent_id_sep', DEFAULT_ENT_ID_SEP)
        else:
            self.add_patterns(cfg)
        return self

    def to_bytes(self, *, exclude: Iterable[str]=SimpleFrozenList()) -> bytes:
        if False:
            i = 10
            return i + 15
        'Serialize the entity ruler patterns to a bytestring.\n\n        RETURNS (bytes): The serialized patterns.\n\n        DOCS: https://spacy.io/api/entityruler#to_bytes\n        '
        serial = {'overwrite': self.overwrite, 'ent_id_sep': self.ent_id_sep, 'phrase_matcher_attr': self.phrase_matcher_attr, 'patterns': self.patterns}
        return srsly.msgpack_dumps(serial)

    def from_disk(self, path: Union[str, Path], *, exclude: Iterable[str]=SimpleFrozenList()) -> 'EntityRuler':
        if False:
            while True:
                i = 10
        'Load the entity ruler from a file. Expects a file containing\n        newline-delimited JSON (JSONL) with one entry per line.\n\n        path (str / Path): The JSONL file to load.\n        RETURNS (EntityRuler): The loaded entity ruler.\n\n        DOCS: https://spacy.io/api/entityruler#from_disk\n        '
        path = ensure_path(path)
        self.clear()
        depr_patterns_path = path.with_suffix('.jsonl')
        if path.suffix == '.jsonl':
            if path.is_file:
                patterns = srsly.read_jsonl(path)
                self.add_patterns(patterns)
            else:
                raise ValueError(Errors.E1023.format(path=path))
        elif depr_patterns_path.is_file():
            patterns = srsly.read_jsonl(depr_patterns_path)
            self.add_patterns(patterns)
        elif path.is_dir():
            cfg = {}
            deserializers_patterns = {'patterns': lambda p: self.add_patterns(srsly.read_jsonl(p.with_suffix('.jsonl')))}
            deserializers_cfg = {'cfg': lambda p: cfg.update(srsly.read_json(p))}
            from_disk(path, deserializers_cfg, {})
            self.overwrite = cfg.get('overwrite', False)
            self.phrase_matcher_attr = cfg.get('phrase_matcher_attr')
            self.ent_id_sep = cfg.get('ent_id_sep', DEFAULT_ENT_ID_SEP)
            self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr=self.phrase_matcher_attr)
            from_disk(path, deserializers_patterns, {})
        else:
            raise ValueError(Errors.E146.format(path=path))
        return self

    def to_disk(self, path: Union[str, Path], *, exclude: Iterable[str]=SimpleFrozenList()) -> None:
        if False:
            while True:
                i = 10
        'Save the entity ruler patterns to a directory. The patterns will be\n        saved as newline-delimited JSON (JSONL).\n\n        path (str / Path): The JSONL file to save.\n\n        DOCS: https://spacy.io/api/entityruler#to_disk\n        '
        path = ensure_path(path)
        cfg = {'overwrite': self.overwrite, 'phrase_matcher_attr': self.phrase_matcher_attr, 'ent_id_sep': self.ent_id_sep}
        serializers = {'patterns': lambda p: srsly.write_jsonl(p.with_suffix('.jsonl'), self.patterns), 'cfg': lambda p: srsly.write_json(p, cfg)}
        if path.suffix == '.jsonl':
            srsly.write_jsonl(path, self.patterns)
        else:
            to_disk(path, serializers, {})