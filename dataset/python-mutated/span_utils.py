from typing import Callable, List, Set, Tuple, TypeVar, Optional
import warnings
from allennlp.common.checks import ConfigurationError
from allennlp.data.tokenizers import Token
TypedSpan = Tuple[int, Tuple[int, int]]
TypedStringSpan = Tuple[str, Tuple[int, int]]

class InvalidTagSequence(Exception):

    def __init__(self, tag_sequence=None):
        if False:
            print('Hello World!')
        super().__init__()
        self.tag_sequence = tag_sequence

    def __str__(self):
        if False:
            while True:
                i = 10
        return ' '.join(self.tag_sequence)
T = TypeVar('T', str, Token)

def enumerate_spans(sentence: List[T], offset: int=0, max_span_width: int=None, min_span_width: int=1, filter_function: Callable[[List[T]], bool]=None) -> List[Tuple[int, int]]:
    if False:
        while True:
            i = 10
    '\n    Given a sentence, return all token spans within the sentence. Spans are `inclusive`.\n    Additionally, you can provide a maximum and minimum span width, which will be used\n    to exclude spans outside of this range.\n\n    Finally, you can provide a function mapping `List[T] -> bool`, which will\n    be applied to every span to decide whether that span should be included. This\n    allows filtering by length, regex matches, pos tags or any Spacy `Token`\n    attributes, for example.\n\n    # Parameters\n\n    sentence : `List[T]`, required.\n        The sentence to generate spans for. The type is generic, as this function\n        can be used with strings, or Spacy `Tokens` or other sequences.\n    offset : `int`, optional (default = `0`)\n        A numeric offset to add to all span start and end indices. This is helpful\n        if the sentence is part of a larger structure, such as a document, which\n        the indices need to respect.\n    max_span_width : `int`, optional (default = `None`)\n        The maximum length of spans which should be included. Defaults to len(sentence).\n    min_span_width : `int`, optional (default = `1`)\n        The minimum length of spans which should be included. Defaults to 1.\n    filter_function : `Callable[[List[T]], bool]`, optional (default = `None`)\n        A function mapping sequences of the passed type T to a boolean value.\n        If `True`, the span is included in the returned spans from the\n        sentence, otherwise it is excluded..\n    '
    max_span_width = max_span_width or len(sentence)
    filter_function = filter_function or (lambda x: True)
    spans: List[Tuple[int, int]] = []
    for start_index in range(len(sentence)):
        last_end_index = min(start_index + max_span_width, len(sentence))
        first_end_index = min(start_index + min_span_width - 1, len(sentence))
        for end_index in range(first_end_index, last_end_index):
            start = offset + start_index
            end = offset + end_index
            if filter_function(sentence[slice(start_index, end_index + 1)]):
                spans.append((start, end))
    return spans

def bio_tags_to_spans(tag_sequence: List[str], classes_to_ignore: List[str]=None) -> List[TypedStringSpan]:
    if False:
        while True:
            i = 10
    '\n    Given a sequence corresponding to BIO tags, extracts spans.\n    Spans are inclusive and can be of zero length, representing a single word span.\n    Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),\n    as otherwise it is possible to get a perfect precision score whilst still predicting\n    ill-formed spans in addition to the correct spans. This function works properly when\n    the spans are unlabeled (i.e., your labels are simply "B", "I", and "O").\n\n    # Parameters\n\n    tag_sequence : `List[str]`, required.\n        The integer class labels for a sequence.\n    classes_to_ignore : `List[str]`, optional (default = `None`).\n        A list of string class labels `excluding` the bio tag\n        which should be ignored when extracting spans.\n\n    # Returns\n\n    spans : `List[TypedStringSpan]`\n        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).\n        Note that the label `does not` contain any BIO tag prefixes.\n    '
    classes_to_ignore = classes_to_ignore or []
    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    span_start = 0
    span_end = 0
    active_conll_tag = None
    for (index, string_tag) in enumerate(tag_sequence):
        bio_tag = string_tag[0]
        if bio_tag not in ['B', 'I', 'O']:
            raise InvalidTagSequence(tag_sequence)
        conll_tag = string_tag[2:]
        if bio_tag == 'O' or conll_tag in classes_to_ignore:
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
            continue
        elif bio_tag == 'B':
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
        elif bio_tag == 'I' and conll_tag == active_conll_tag:
            span_end += 1
        else:
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)

def iob1_tags_to_spans(tag_sequence: List[str], classes_to_ignore: List[str]=None) -> List[TypedStringSpan]:
    if False:
        while True:
            i = 10
    '\n    Given a sequence corresponding to IOB1 tags, extracts spans.\n    Spans are inclusive and can be of zero length, representing a single word span.\n    Ill-formed spans are also included (i.e., those where "B-LABEL" is not preceded\n    by "I-LABEL" or "B-LABEL").\n\n    # Parameters\n\n    tag_sequence : `List[str]`, required.\n        The integer class labels for a sequence.\n    classes_to_ignore : `List[str]`, optional (default = `None`).\n        A list of string class labels `excluding` the bio tag\n        which should be ignored when extracting spans.\n\n    # Returns\n\n    spans : `List[TypedStringSpan]`\n        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).\n        Note that the label `does not` contain any BIO tag prefixes.\n    '
    classes_to_ignore = classes_to_ignore or []
    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    span_start = 0
    span_end = 0
    active_conll_tag = None
    prev_bio_tag = None
    prev_conll_tag = None
    for (index, string_tag) in enumerate(tag_sequence):
        curr_bio_tag = string_tag[0]
        curr_conll_tag = string_tag[2:]
        if curr_bio_tag not in ['B', 'I', 'O']:
            raise InvalidTagSequence(tag_sequence)
        if curr_bio_tag == 'O' or curr_conll_tag in classes_to_ignore:
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
        elif _iob1_start_of_chunk(prev_bio_tag, prev_conll_tag, curr_bio_tag, curr_conll_tag):
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = curr_conll_tag
            span_start = index
            span_end = index
        else:
            span_end += 1
        prev_bio_tag = string_tag[0]
        prev_conll_tag = string_tag[2:]
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)

def _iob1_start_of_chunk(prev_bio_tag: Optional[str], prev_conll_tag: Optional[str], curr_bio_tag: str, curr_conll_tag: str) -> bool:
    if False:
        i = 10
        return i + 15
    if curr_bio_tag == 'B':
        return True
    if curr_bio_tag == 'I' and prev_bio_tag == 'O':
        return True
    if curr_bio_tag != 'O' and prev_conll_tag != curr_conll_tag:
        return True
    return False

def bioul_tags_to_spans(tag_sequence: List[str], classes_to_ignore: List[str]=None) -> List[TypedStringSpan]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a sequence corresponding to BIOUL tags, extracts spans.\n    Spans are inclusive and can be of zero length, representing a single word span.\n    Ill-formed spans are not allowed and will raise `InvalidTagSequence`.\n    This function works properly when the spans are unlabeled (i.e., your labels are\n    simply "B", "I", "O", "U", and "L").\n\n    # Parameters\n\n    tag_sequence : `List[str]`, required.\n        The tag sequence encoded in BIOUL, e.g. ["B-PER", "L-PER", "O"].\n    classes_to_ignore : `List[str]`, optional (default = `None`).\n        A list of string class labels `excluding` the bio tag\n        which should be ignored when extracting spans.\n\n    # Returns\n\n    spans : `List[TypedStringSpan]`\n        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).\n    '
    spans = []
    classes_to_ignore = classes_to_ignore or []
    index = 0
    while index < len(tag_sequence):
        label = tag_sequence[index]
        if label[0] == 'U':
            spans.append((label.partition('-')[2], (index, index)))
        elif label[0] == 'B':
            start = index
            while label[0] != 'L':
                index += 1
                if index >= len(tag_sequence):
                    raise InvalidTagSequence(tag_sequence)
                label = tag_sequence[index]
                if not (label[0] == 'I' or label[0] == 'L'):
                    raise InvalidTagSequence(tag_sequence)
            spans.append((label.partition('-')[2], (start, index)))
        elif label != 'O':
            raise InvalidTagSequence(tag_sequence)
        index += 1
    return [span for span in spans if span[0] not in classes_to_ignore]

def iob1_to_bioul(tag_sequence: List[str]) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    warnings.warn("iob1_to_bioul has been replaced with 'to_bioul' to allow more encoding options.", FutureWarning)
    return to_bioul(tag_sequence)

def to_bioul(tag_sequence: List[str], encoding: str='IOB1') -> List[str]:
    if False:
        while True:
            i = 10
    '\n    Given a tag sequence encoded with IOB1 labels, recode to BIOUL.\n\n    In the IOB1 scheme, I is a token inside a span, O is a token outside\n    a span and B is the beginning of span immediately following another\n    span of the same type.\n\n    In the BIO scheme, I is a token inside a span, O is a token outside\n    a span and B is the beginning of a span.\n\n    # Parameters\n\n    tag_sequence : `List[str]`, required.\n        The tag sequence encoded in IOB1, e.g. ["I-PER", "I-PER", "O"].\n    encoding : `str`, optional, (default = `"IOB1"`).\n        The encoding type to convert from. Must be either "IOB1" or "BIO".\n\n    # Returns\n\n    bioul_sequence : `List[str]`\n        The tag sequence encoded in IOB1, e.g. ["B-PER", "L-PER", "O"].\n    '
    if encoding not in {'IOB1', 'BIO'}:
        raise ConfigurationError(f"Invalid encoding {encoding} passed to 'to_bioul'.")

    def replace_label(full_label, new_label):
        if False:
            for i in range(10):
                print('nop')
        parts = list(full_label.partition('-'))
        parts[0] = new_label
        return ''.join(parts)

    def pop_replace_append(in_stack, out_stack, new_label):
        if False:
            return 10
        tag = in_stack.pop()
        new_tag = replace_label(tag, new_label)
        out_stack.append(new_tag)

    def process_stack(stack, out_stack):
        if False:
            while True:
                i = 10
        if len(stack) == 1:
            pop_replace_append(stack, out_stack, 'U')
        else:
            recoded_stack = []
            pop_replace_append(stack, recoded_stack, 'L')
            while len(stack) >= 2:
                pop_replace_append(stack, recoded_stack, 'I')
            pop_replace_append(stack, recoded_stack, 'B')
            recoded_stack.reverse()
            out_stack.extend(recoded_stack)
    bioul_sequence = []
    stack: List[str] = []
    for label in tag_sequence:
        if label == 'O' and len(stack) == 0:
            bioul_sequence.append(label)
        elif label == 'O' and len(stack) > 0:
            process_stack(stack, bioul_sequence)
            bioul_sequence.append(label)
        elif label[0] == 'I':
            if len(stack) == 0:
                if encoding == 'BIO':
                    raise InvalidTagSequence(tag_sequence)
                stack.append(label)
            else:
                this_type = label.partition('-')[2]
                prev_type = stack[-1].partition('-')[2]
                if this_type == prev_type:
                    stack.append(label)
                else:
                    if encoding == 'BIO':
                        raise InvalidTagSequence(tag_sequence)
                    process_stack(stack, bioul_sequence)
                    stack.append(label)
        elif label[0] == 'B':
            if len(stack) > 0:
                process_stack(stack, bioul_sequence)
            stack.append(label)
        else:
            raise InvalidTagSequence(tag_sequence)
    if len(stack) > 0:
        process_stack(stack, bioul_sequence)
    return bioul_sequence

def bmes_tags_to_spans(tag_sequence: List[str], classes_to_ignore: List[str]=None) -> List[TypedStringSpan]:
    if False:
        i = 10
        return i + 15
    '\n    Given a sequence corresponding to BMES tags, extracts spans.\n    Spans are inclusive and can be of zero length, representing a single word span.\n    Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),\n    as otherwise it is possible to get a perfect precision score whilst still predicting\n    ill-formed spans in addition to the correct spans.\n    This function works properly when the spans are unlabeled (i.e., your labels are\n    simply "B", "M", "E" and "S").\n\n    # Parameters\n\n    tag_sequence : `List[str]`, required.\n        The integer class labels for a sequence.\n    classes_to_ignore : `List[str]`, optional (default = `None`).\n        A list of string class labels `excluding` the bio tag\n        which should be ignored when extracting spans.\n\n    # Returns\n\n    spans : `List[TypedStringSpan]`\n        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).\n        Note that the label `does not` contain any BIO tag prefixes.\n    '

    def extract_bmes_tag_label(text):
        if False:
            for i in range(10):
                print('nop')
        bmes_tag = text[0]
        label = text[2:]
        return (bmes_tag, label)
    spans: List[Tuple[str, List[int]]] = []
    prev_bmes_tag: Optional[str] = None
    for (index, tag) in enumerate(tag_sequence):
        (bmes_tag, label) = extract_bmes_tag_label(tag)
        if bmes_tag in ('B', 'S'):
            spans.append((label, [index, index]))
        elif bmes_tag in ('M', 'E') and prev_bmes_tag in ('B', 'M') and (spans[-1][0] == label):
            spans[-1][1][1] = index
        else:
            spans.append((label, [index, index]))
        prev_bmes_tag = bmes_tag
    classes_to_ignore = classes_to_ignore or []
    return [(span[0], (span[1][0], span[1][1])) for span in spans if span[0] not in classes_to_ignore]