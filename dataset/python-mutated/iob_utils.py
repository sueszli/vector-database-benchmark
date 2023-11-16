import warnings
from typing import Dict, Iterable, Iterator, List, Tuple, Union, cast
from ..errors import Errors, Warnings
from ..tokens import Doc, Span

def iob_to_biluo(tags: Iterable[str]) -> List[str]:
    if False:
        print('Hello World!')
    out: List[str] = []
    tags = list(tags)
    while tags:
        out.extend(_consume_os(tags))
        out.extend(_consume_ent(tags))
    return out

def biluo_to_iob(tags: Iterable[str]) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    out = []
    for tag in tags:
        if tag is None:
            out.append(tag)
        else:
            tag = tag.replace('U-', 'B-', 1).replace('L-', 'I-', 1)
            out.append(tag)
    return out

def _consume_os(tags: List[str]) -> Iterator[str]:
    if False:
        return 10
    while tags and tags[0] == 'O':
        yield tags.pop(0)

def _consume_ent(tags: List[str]) -> List[str]:
    if False:
        print('Hello World!')
    if not tags:
        return []
    tag = tags.pop(0)
    target_in = 'I' + tag[1:]
    target_last = 'L' + tag[1:]
    length = 1
    while tags and tags[0] in {target_in, target_last}:
        length += 1
        tags.pop(0)
    label = tag[2:]
    if length == 1:
        if len(label) == 0:
            raise ValueError(Errors.E177.format(tag=tag))
        return ['U-' + label]
    else:
        start = 'B-' + label
        end = 'L-' + label
        middle = [f'I-{label}' for _ in range(1, length - 1)]
        return [start] + middle + [end]

def doc_to_biluo_tags(doc: Doc, missing: str='O'):
    if False:
        print('Hello World!')
    return offsets_to_biluo_tags(doc, [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents], missing=missing)

def _doc_to_biluo_tags_with_partial(doc: Doc) -> List[str]:
    if False:
        return 10
    ents = doc_to_biluo_tags(doc, missing='-')
    for (i, token) in enumerate(doc):
        if token.ent_iob == 2:
            ents[i] = 'O'
    return ents

def offsets_to_biluo_tags(doc: Doc, entities: Iterable[Tuple[int, int, Union[str, int]]], missing: str='O') -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Encode labelled spans into per-token tags, using the\n    Begin/In/Last/Unit/Out scheme (BILUO).\n\n    doc (Doc): The document that the entity offsets refer to. The output tags\n        will refer to the token boundaries within the document.\n    entities (iterable): A sequence of `(start, end, label)` triples. `start`\n        and `end` should be character-offset integers denoting the slice into\n        the original string.\n    missing (str): The label used for missing values, e.g. if tokenization\n        doesn’t align with the entity offsets. Defaults to "O".\n    RETURNS (list): A list of unicode strings, describing the tags. Each tag\n        string will be of the form either "", "O" or "{action}-{label}", where\n        action is one of "B", "I", "L", "U". The missing label is used where the\n        entity offsets don\'t align with the tokenization in the `Doc` object.\n        The training algorithm will view these as missing values. "O" denotes a\n        non-entity token. "B" denotes the beginning of a multi-token entity,\n        "I" the inside of an entity of three or more tokens, and "L" the end\n        of an entity of two or more tokens. "U" denotes a single-token entity.\n\n    EXAMPLE:\n        >>> text = \'I like London.\'\n        >>> entities = [(len(\'I like \'), len(\'I like London\'), \'LOC\')]\n        >>> doc = nlp.tokenizer(text)\n        >>> tags = offsets_to_biluo_tags(doc, entities)\n        >>> assert tags == ["O", "O", \'U-LOC\', "O"]\n    '
    tokens_in_ents: Dict[int, Tuple[int, int, Union[str, int]]] = {}
    starts = {token.idx: token.i for token in doc}
    ends = {token.idx + len(token): token.i for token in doc}
    biluo = ['-' for _ in doc]
    for (start_char, end_char, label) in entities:
        if not label:
            for s in starts:
                if s >= start_char and s < end_char:
                    biluo[starts[s]] = 'O'
        else:
            for token_index in range(start_char, end_char):
                if token_index in tokens_in_ents.keys():
                    raise ValueError(Errors.E103.format(span1=(tokens_in_ents[token_index][0], tokens_in_ents[token_index][1], tokens_in_ents[token_index][2]), span2=(start_char, end_char, label)))
                tokens_in_ents[token_index] = (start_char, end_char, label)
            start_token = starts.get(start_char)
            end_token = ends.get(end_char)
            if start_token is not None and end_token is not None:
                if start_token == end_token:
                    biluo[start_token] = f'U-{label}'
                else:
                    biluo[start_token] = f'B-{label}'
                    for i in range(start_token + 1, end_token):
                        biluo[i] = f'I-{label}'
                    biluo[end_token] = f'L-{label}'
    entity_chars = set()
    for (start_char, end_char, label) in entities:
        for i in range(start_char, end_char):
            entity_chars.add(i)
    for token in doc:
        for i in range(token.idx, token.idx + len(token)):
            if i in entity_chars:
                break
        else:
            biluo[token.i] = missing
    if '-' in biluo and missing != '-':
        ent_str = str(entities)
        warnings.warn(Warnings.W030.format(text=doc.text[:50] + '...' if len(doc.text) > 50 else doc.text, entities=ent_str[:50] + '...' if len(ent_str) > 50 else ent_str))
    return biluo

def biluo_tags_to_spans(doc: Doc, tags: Iterable[str]) -> List[Span]:
    if False:
        i = 10
        return i + 15
    'Encode per-token tags following the BILUO scheme into Span object, e.g.\n    to overwrite the doc.ents.\n\n    doc (Doc): The document that the BILUO tags refer to.\n    tags (iterable): A sequence of BILUO tags with each tag describing one\n        token. Each tag string will be of the form of either "", "O" or\n        "{action}-{label}", where action is one of "B", "I", "L", "U".\n    RETURNS (list): A sequence of Span objects. Each token with a missing IOB\n        tag is returned as a Span with an empty label.\n    '
    token_offsets = tags_to_entities(tags)
    spans = []
    for (label, start_idx, end_idx) in token_offsets:
        span = Span(doc, start_idx, end_idx + 1, label=label)
        spans.append(span)
    return spans

def biluo_tags_to_offsets(doc: Doc, tags: Iterable[str]) -> List[Tuple[int, int, Union[str, int]]]:
    if False:
        while True:
            i = 10
    'Encode per-token tags following the BILUO scheme into entity offsets.\n\n    doc (Doc): The document that the BILUO tags refer to.\n    tags (iterable): A sequence of BILUO tags with each tag describing one\n        token. Each tags string will be of the form of either "", "O" or\n        "{action}-{label}", where action is one of "B", "I", "L", "U".\n    RETURNS (list): A sequence of `(start, end, label)` triples. `start` and\n        `end` will be character-offset integers denoting the slice into the\n        original string.\n    '
    spans = biluo_tags_to_spans(doc, tags)
    return [(span.start_char, span.end_char, span.label_) for span in spans]

def tags_to_entities(tags: Iterable[str]) -> List[Tuple[str, int, int]]:
    if False:
        for i in range(10):
            print('nop')
    'Note that the end index returned by this function is inclusive.\n    To use it for Span creation, increment the end by 1.'
    entities = []
    start = None
    for (i, tag) in enumerate(tags):
        if tag is None or tag.startswith('-'):
            if start is not None:
                start = None
            else:
                entities.append(('', i, i))
        elif tag.startswith('O'):
            pass
        elif tag.startswith('I'):
            if start is None:
                raise ValueError(Errors.E067.format(start='I', tags=list(tags)[:i + 1]))
        elif tag.startswith('U'):
            entities.append((tag[2:], i, i))
        elif tag.startswith('B'):
            start = i
        elif tag.startswith('L'):
            if start is None:
                raise ValueError(Errors.E067.format(start='L', tags=list(tags)[:i + 1]))
            entities.append((tag[2:], start, i))
            start = None
        else:
            raise ValueError(Errors.E068.format(tag=tag))
    return entities

def split_bilu_label(label: str) -> Tuple[str, str]:
    if False:
        while True:
            i = 10
    return cast(Tuple[str, str], label.split('-', 1))

def remove_bilu_prefix(label: str) -> str:
    if False:
        i = 10
        return i + 15
    return label.split('-', 1)[1]
offsets_from_biluo_tags = biluo_tags_to_offsets
spans_from_biluo_tags = biluo_tags_to_spans
biluo_tags_from_offsets = offsets_to_biluo_tags