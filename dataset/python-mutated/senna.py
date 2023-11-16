"""
Senna POS tagger, NER Tagger, Chunk Tagger

The input is:

- path to the directory that contains SENNA executables. If the path is incorrect,
  SennaTagger will automatically search for executable file specified in SENNA environment variable
- (optionally) the encoding of the input data (default:utf-8)

Note: Unit tests for this module can be found in test/unit/test_senna.py

>>> from nltk.tag import SennaTagger
>>> tagger = SennaTagger('/usr/share/senna-v3.0')  # doctest: +SKIP
>>> tagger.tag('What is the airspeed of an unladen swallow ?'.split()) # doctest: +SKIP
[('What', 'WP'), ('is', 'VBZ'), ('the', 'DT'), ('airspeed', 'NN'),
('of', 'IN'), ('an', 'DT'), ('unladen', 'NN'), ('swallow', 'NN'), ('?', '.')]

>>> from nltk.tag import SennaChunkTagger
>>> chktagger = SennaChunkTagger('/usr/share/senna-v3.0')  # doctest: +SKIP
>>> chktagger.tag('What is the airspeed of an unladen swallow ?'.split()) # doctest: +SKIP
[('What', 'B-NP'), ('is', 'B-VP'), ('the', 'B-NP'), ('airspeed', 'I-NP'),
('of', 'B-PP'), ('an', 'B-NP'), ('unladen', 'I-NP'), ('swallow', 'I-NP'),
('?', 'O')]

>>> from nltk.tag import SennaNERTagger
>>> nertagger = SennaNERTagger('/usr/share/senna-v3.0')  # doctest: +SKIP
>>> nertagger.tag('Shakespeare theatre was in London .'.split()) # doctest: +SKIP
[('Shakespeare', 'B-PER'), ('theatre', 'O'), ('was', 'O'), ('in', 'O'),
('London', 'B-LOC'), ('.', 'O')]
>>> nertagger.tag('UN headquarters are in NY , USA .'.split()) # doctest: +SKIP
[('UN', 'B-ORG'), ('headquarters', 'O'), ('are', 'O'), ('in', 'O'),
('NY', 'B-LOC'), (',', 'O'), ('USA', 'B-LOC'), ('.', 'O')]
"""
from nltk.classify import Senna

class SennaTagger(Senna):

    def __init__(self, path, encoding='utf-8'):
        if False:
            i = 10
            return i + 15
        super().__init__(path, ['pos'], encoding)

    def tag_sents(self, sentences):
        if False:
            return 10
        '\n        Applies the tag method over a list of sentences. This method will return\n        for each sentence a list of tuples of (word, tag).\n        '
        tagged_sents = super().tag_sents(sentences)
        for i in range(len(tagged_sents)):
            for j in range(len(tagged_sents[i])):
                annotations = tagged_sents[i][j]
                tagged_sents[i][j] = (annotations['word'], annotations['pos'])
        return tagged_sents

class SennaChunkTagger(Senna):

    def __init__(self, path, encoding='utf-8'):
        if False:
            i = 10
            return i + 15
        super().__init__(path, ['chk'], encoding)

    def tag_sents(self, sentences):
        if False:
            while True:
                i = 10
        '\n        Applies the tag method over a list of sentences. This method will return\n        for each sentence a list of tuples of (word, tag).\n        '
        tagged_sents = super().tag_sents(sentences)
        for i in range(len(tagged_sents)):
            for j in range(len(tagged_sents[i])):
                annotations = tagged_sents[i][j]
                tagged_sents[i][j] = (annotations['word'], annotations['chk'])
        return tagged_sents

    def bio_to_chunks(self, tagged_sent, chunk_type):
        if False:
            i = 10
            return i + 15
        "\n        Extracts the chunks in a BIO chunk-tagged sentence.\n\n        >>> from nltk.tag import SennaChunkTagger\n        >>> chktagger = SennaChunkTagger('/usr/share/senna-v3.0')  # doctest: +SKIP\n        >>> sent = 'What is the airspeed of an unladen swallow ?'.split()\n        >>> tagged_sent = chktagger.tag(sent)  # doctest: +SKIP\n        >>> tagged_sent  # doctest: +SKIP\n        [('What', 'B-NP'), ('is', 'B-VP'), ('the', 'B-NP'), ('airspeed', 'I-NP'),\n        ('of', 'B-PP'), ('an', 'B-NP'), ('unladen', 'I-NP'), ('swallow', 'I-NP'),\n        ('?', 'O')]\n        >>> list(chktagger.bio_to_chunks(tagged_sent, chunk_type='NP'))  # doctest: +SKIP\n        [('What', '0'), ('the airspeed', '2-3'), ('an unladen swallow', '5-6-7')]\n\n        :param tagged_sent: A list of tuples of word and BIO chunk tag.\n        :type tagged_sent: list(tuple)\n        :param tagged_sent: The chunk tag that users want to extract, e.g. 'NP' or 'VP'\n        :type tagged_sent: str\n\n        :return: An iterable of tuples of chunks that users want to extract\n          and their corresponding indices.\n        :rtype: iter(tuple(str))\n        "
        current_chunk = []
        current_chunk_position = []
        for (idx, word_pos) in enumerate(tagged_sent):
            (word, pos) = word_pos
            if '-' + chunk_type in pos:
                current_chunk.append(word)
                current_chunk_position.append(idx)
            elif current_chunk:
                _chunk_str = ' '.join(current_chunk)
                _chunk_pos_str = '-'.join(map(str, current_chunk_position))
                yield (_chunk_str, _chunk_pos_str)
                current_chunk = []
                current_chunk_position = []
        if current_chunk:
            yield (' '.join(current_chunk), '-'.join(map(str, current_chunk_position)))

class SennaNERTagger(Senna):

    def __init__(self, path, encoding='utf-8'):
        if False:
            while True:
                i = 10
        super().__init__(path, ['ner'], encoding)

    def tag_sents(self, sentences):
        if False:
            print('Hello World!')
        '\n        Applies the tag method over a list of sentences. This method will return\n        for each sentence a list of tuples of (word, tag).\n        '
        tagged_sents = super().tag_sents(sentences)
        for i in range(len(tagged_sents)):
            for j in range(len(tagged_sents[i])):
                annotations = tagged_sents[i][j]
                tagged_sents[i][j] = (annotations['word'], annotations['ner'])
        return tagged_sents