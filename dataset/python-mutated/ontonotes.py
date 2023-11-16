from typing import Dict, DefaultDict, List, Optional, Iterator, Set, Tuple, Union
from collections import defaultdict
import codecs
import os
import logging
from nltk import Tree
import tqdm
logger = logging.getLogger(__name__)
TypedSpan = Tuple[Union[int, str], Tuple[int, int]]

class OntonotesSentence:
    """
    A class representing the annotations available for a single CONLL formatted sentence.

    Parameters
    ----------
    document_id : ``str``
        This is a variation on the document filename
    sentence_id : ``int``
        The integer ID of the sentence within a document.
    words : ``List[str]``
        This is the tokens as segmented/tokenized in the Treebank.
    pos_tags : ``List[str]``
        This is the Penn-Treebank-style part of speech. When parse information is missing,
        all parts of speech except the one for which there is some sense or proposition
        annotation are marked with a XX tag. The verb is marked with just a VERB tag.
    parse_tree : ``nltk.Tree``
        An nltk Tree representing the parse. It does not include POS tags. When the parse information
        is missing, the parse will have a single node, e.g ``(TOP all of the words ...)``.
    predicate_lemmas : ``List[Optional[str]]``
        The predicate lemma of the words for which we have semantic role
        information or word sense information. All other indices are ``None``.
    predicate_framenet_ids : ``List[Optional[int]]``
        The PropBank frameset ID of the lemmas in ``predicate_lemmas``, or ``None``.
    word_senses : ``List[int]``
        The word senses for the words in the sentence, or ``None``.
    speakers : ``List[Optional[str]]``
        The speaker information for the words in the sentence, if present, or ``None``
        This is the speaker or author name where available. Mostly in Broadcast Conversation
        and Web Log data. When not available the rows are marked with an "-".
    named_entities : ``List[str]``
        The BIO tags for named entities in the sentence.
    srl_frames : ``Dict[str, List[str]]``
        A dictionary keyed by the verb in the sentence for the given
        Propbank frame labels, in a BIO format.
    coref_spans : ``Set[TypedSpan]``
        The spans for entity mentions involved in coreference resolution within the sentence.
        Each element is a tuple composed of (cluster_id, (start_index, end_index)). Indices
        are `inclusive`.
    """

    def __init__(self, document_id: str, sentence_id: int, words: List[str], pos_tags: List[str], parse_tree: Tree, predicate_lemmas: List[Optional[str]], predicate_framenet_ids: List[Optional[str]], word_senses: List[Optional[int]], speakers: List[Optional[str]], named_entities: List[str], srl_frames: Dict[str, List[str]], coref_spans: Set[TypedSpan]) -> None:
        if False:
            return 10
        self.document_id = document_id
        self.sentence_id = sentence_id
        self.words = words
        self.pos_tags = pos_tags
        self.parse_tree = parse_tree
        self.predicate_lemmas = predicate_lemmas
        self.predicate_framenet_ids = predicate_framenet_ids
        self.word_senses = word_senses
        self.speakers = speakers
        self.named_entities = named_entities
        self.srl_frames = srl_frames
        self.coref_spans = coref_spans

class Ontonotes:
    """
    This DatasetReader is designed to read in the English OntoNotes v5.0 data
    in the format used by the CoNLL 2011/2012 shared tasks. In order to use this
    Reader, you must follow the instructions provided `here (v12 release):
    <http://cemantix.org/data/ontonotes.html>`_, which will allow you to download
    the CoNLL style annotations for the  OntoNotes v5.0 release -- LDC2013T19.tgz
    obtained from LDC.

    Once you have run the scripts on the extracted data, you will have a folder
    structured as follows:

    conll-formatted-ontonotes-5.0/
     ── data
       ├── development
           └── data
               └── english
                   └── annotations
                       ├── bc
                       ├── bn
                       ├── mz
                       ├── nw
                       ├── pt
                       ├── tc
                       └── wb
       ├── test
           └── data
               └── english
                   └── annotations
                       ├── bc
                       ├── bn
                       ├── mz
                       ├── nw
                       ├── pt
                       ├── tc
                       └── wb
       └── train
           └── data
               └── english
                   └── annotations
                       ├── bc
                       ├── bn
                       ├── mz
                       ├── nw
                       ├── pt
                       ├── tc
                       └── wb

    The file path provided to this class can then be any of the train, test or development
    directories(or the top level data directory, if you are not utilizing the splits).

    The data has the following format, ordered by column.

    1 Document ID : ``str``
        This is a variation on the document filename
    2 Part number : ``int``
        Some files are divided into multiple parts numbered as 000, 001, 002, ... etc.
    3 Word number : ``int``
        This is the word index of the word in that sentence.
    4 Word : ``str``
        This is the token as segmented/tokenized in the Treebank. Initially the ``*_skel`` file
        contain the placeholder [WORD] which gets replaced by the actual token from the
        Treebank which is part of the OntoNotes release.
    5 POS Tag : ``str``
        This is the Penn Treebank style part of speech. When parse information is missing,
        all part of speeches except the one for which there is some sense or proposition
        annotation are marked with a XX tag. The verb is marked with just a VERB tag.
    6 Parse bit: ``str``
        This is the bracketed structure broken before the first open parenthesis in the parse,
        and the word/part-of-speech leaf replaced with a ``*``. When the parse information is
        missing, the first word of a sentence is tagged as ``(TOP*`` and the last word is tagged
        as ``*)`` and all intermediate words are tagged with a ``*``.
    7 Predicate lemma: ``str``
        The predicate lemma is mentioned for the rows for which we have semantic role
        information or word sense information. All other rows are marked with a "-".
    8 Predicate Frameset ID: ``int``
        The PropBank frameset ID of the predicate in Column 7.
    9 Word sense: ``float``
        This is the word sense of the word in Column 3.
    10 Speaker/Author: ``str``
        This is the speaker or author name where available. Mostly in Broadcast Conversation
        and Web Log data. When not available the rows are marked with an "-".
    11 Named Entities: ``str``
        These columns identifies the spans representing various named entities. For documents
        which do not have named entity annotation, each line is represented with an ``*``.
    12+ Predicate Arguments: ``str``
        There is one column each of predicate argument structure information for the predicate
        mentioned in Column 7. If there are no predicates tagged in a sentence this is a
        single column with all rows marked with an ``*``.
    -1 Co-reference: ``str``
        Co-reference chain information encoded in a parenthesis structure. For documents that do
         not have co-reference annotations, each line is represented with a "-".
    """

    def dataset_iterator(self, file_path) -> Iterator[OntonotesSentence]:
        if False:
            return 10
        '\n        An iterator over the entire dataset, yielding all sentences processed.\n        '
        for conll_file in self.dataset_path_iterator(file_path):
            yield from self.sentence_iterator(conll_file)

    @staticmethod
    def dataset_path_iterator(file_path: str) -> Iterator[str]:
        if False:
            while True:
                i = 10
        '\n        An iterator returning file_paths in a directory\n        containing CONLL-formatted files.\n        '
        logger.info('Reading CONLL sentences from dataset files at: %s', file_path)
        for (root, _, files) in tqdm.tqdm(list(os.walk(file_path))):
            for data_file in files:
                if not data_file.endswith('gold_conll'):
                    continue
                yield os.path.join(root, data_file)

    def sentence_iterator(self, file_path: str) -> Iterator[OntonotesSentence]:
        if False:
            return 10
        '\n        An iterator over the sentences in an individual CONLL formatted file.\n        '
        with codecs.open(file_path, 'r', encoding='utf8') as open_file:
            conll_rows = []
            for line in open_file:
                line = line.strip()
                if line != '' and (not line.startswith('#')):
                    conll_rows.append(line)
                elif not conll_rows:
                    continue
                else:
                    yield self._conll_rows_to_sentence(conll_rows)
                    conll_rows = []

    def _conll_rows_to_sentence(self, conll_rows: List[str]) -> OntonotesSentence:
        if False:
            return 10
        document_id: str = None
        sentence_id: int = None
        sentence: List[str] = []
        pos_tags: List[str] = []
        parse_pieces: List[str] = []
        predicate_lemmas: List[str] = []
        predicate_framenet_ids: List[str] = []
        word_senses: List[int] = []
        speakers: List[str] = []
        verbal_predicates: List[str] = []
        span_labels: List[List[str]] = []
        current_span_labels: List[str] = []
        clusters: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)
        coref_stacks: DefaultDict[int, List[int]] = defaultdict(list)
        for (index, row) in enumerate(conll_rows):
            conll_components = row.split()
            document_id = conll_components[0]
            sentence_id = int(conll_components[1])
            word = conll_components[3]
            pos_tag = conll_components[4]
            parse_piece = conll_components[5]
            if word == '(':
                parse_word = '-LRB-'
            elif word == ')':
                parse_word = '-RRB-'
            else:
                parse_word = word
            parse_piece = parse_piece.replace('*', f' {parse_word}')
            lemmatised_word = conll_components[6]
            framenet_id = conll_components[7]
            word_sense = conll_components[8]
            speaker = conll_components[9]
            if not span_labels:
                span_labels = [[] for _ in conll_components[10:-1]]
                current_span_labels = [None for _ in conll_components[10:-1]]
            self._process_span_annotations_for_word(conll_components[10:-1], span_labels, current_span_labels)
            word_is_verbal_predicate = any(['(V' in x for x in conll_components[11:-1]])
            if word_is_verbal_predicate:
                verbal_predicates.append(word)
            self._process_coref_span_annotations_for_word(conll_components[-1], index, clusters, coref_stacks)
            sentence.append(word)
            pos_tags.append(pos_tag)
            parse_pieces.append(parse_piece)
            predicate_lemmas.append(lemmatised_word if lemmatised_word != '-' else None)
            predicate_framenet_ids.append(framenet_id if framenet_id != '-' else None)
            word_senses.append(int(word_sense) if word_sense != '-' else None)
            speakers.append(speaker if speaker != '-' else None)
        named_entities = span_labels[0]
        srl_frames = {predicate: labels for (predicate, labels) in zip(verbal_predicates, span_labels[1:])}
        parse_tree = Tree.fromstring(''.join(parse_pieces))
        coref_span_tuples: Set[TypedSpan] = {(cluster_id, span) for (cluster_id, span_list) in clusters.items() for span in span_list}
        return OntonotesSentence(document_id, sentence_id, sentence, pos_tags, parse_tree, predicate_lemmas, predicate_framenet_ids, word_senses, speakers, named_entities, srl_frames, coref_span_tuples)

    @staticmethod
    def _process_coref_span_annotations_for_word(label: str, word_index: int, clusters: DefaultDict[int, List[Tuple[int, int]]], coref_stacks: DefaultDict[int, List[int]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        For a given coref label, add it to a currently open span(s), complete a span(s) or\n        ignore it, if it is outside of all spans. This method mutates the clusters and coref_stacks\n        dictionaries.\n\n        Parameters\n        ----------\n        label : ``str``\n            The coref label for this word.\n        word_index : ``int``\n            The word index into the sentence.\n        clusters : ``DefaultDict[int, List[Tuple[int, int]]]``\n            A dictionary mapping cluster ids to lists of inclusive spans into the\n            sentence.\n        coref_stacks: ``DefaultDict[int, List[int]]``\n            Stacks for each cluster id to hold the start indices of active spans (spans\n            which we are inside of when processing a given word). Spans with the same id\n            can be nested, which is why we collect these opening spans on a stack, e.g:\n\n            [Greg, the baker who referred to [himself]_ID1 as 'the bread man']_ID1\n        "
        if label != '-':
            for segment in label.split('|'):
                if segment[0] == '(':
                    if segment[-1] == ')':
                        cluster_id = int(segment[1:-1])
                        clusters[cluster_id].append((word_index, word_index))
                    else:
                        cluster_id = int(segment[1:])
                        coref_stacks[cluster_id].append(word_index)
                else:
                    cluster_id = int(segment[:-1])
                    start = coref_stacks[cluster_id].pop()
                    clusters[cluster_id].append((start, word_index))

    @staticmethod
    def _process_span_annotations_for_word(annotations: List[str], span_labels: List[List[str]], current_span_labels: List[Optional[str]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Given a sequence of different label types for a single word and the current\n        span label we are inside, compute the BIO tag for each label and append to a list.\n\n        Parameters\n        ----------\n        annotations: ``List[str]``\n            A list of labels to compute BIO tags for.\n        span_labels : ``List[List[str]]``\n            A list of lists, one for each annotation, to incrementally collect\n            the BIO tags for a sequence.\n        current_span_labels : ``List[Optional[str]]``\n            The currently open span per annotation type, or ``None`` if there is no open span.\n        '
        for (annotation_index, annotation) in enumerate(annotations):
            label = annotation.strip('()*')
            if '(' in annotation:
                bio_label = 'B-' + label
                span_labels[annotation_index].append(bio_label)
                current_span_labels[annotation_index] = label
            elif current_span_labels[annotation_index] is not None:
                bio_label = 'I-' + current_span_labels[annotation_index]
                span_labels[annotation_index].append(bio_label)
            else:
                span_labels[annotation_index].append('O')
            if ')' in annotation:
                current_span_labels[annotation_index] = None