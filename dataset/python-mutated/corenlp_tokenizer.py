"""Simple wrapper around the Stanford CoreNLP pipeline.

Serves commands to a java subprocess running the jar. Requires java 8.
"""
import copy
import json
import pexpect
from .tokenizer import Tokens, Tokenizer
from . import DEFAULTS

class CoreNLPTokenizer(Tokenizer):

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        '\n        Args:\n            annotators: set that can include pos, lemma, and ner.\n            classpath: Path to the corenlp directory of jars\n            mem: Java heap memory\n        '
        self.classpath = kwargs.get('classpath') or DEFAULTS['corenlp_classpath']
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        self.mem = kwargs.get('mem', '2g')
        self._launch()

    def _launch(self):
        if False:
            return 10
        'Start the CoreNLP jar with pexpect.'
        annotators = ['tokenize', 'ssplit']
        if 'ner' in self.annotators:
            annotators.extend(['pos', 'lemma', 'ner'])
        elif 'lemma' in self.annotators:
            annotators.extend(['pos', 'lemma'])
        elif 'pos' in self.annotators:
            annotators.extend(['pos'])
        annotators = ','.join(annotators)
        options = ','.join(['untokenizable=noneDelete', 'invertible=true'])
        cmd = ['java', '-mx' + self.mem, '-cp', '"%s"' % self.classpath, 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', annotators, '-tokenize.options', options, '-outputFormat', 'json', '-prettyPrint', 'false']
        self.corenlp = pexpect.spawn('/bin/bash', maxread=100000, timeout=60)
        self.corenlp.setecho(False)
        self.corenlp.sendline('stty -icanon')
        self.corenlp.sendline(' '.join(cmd))
        self.corenlp.delaybeforesend = 0
        self.corenlp.delayafterread = 0
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)

    @staticmethod
    def _convert(token):
        if False:
            print('Hello World!')
        if token == '-LRB-':
            return '('
        if token == '-RRB-':
            return ')'
        if token == '-LSB-':
            return '['
        if token == '-RSB-':
            return ']'
        if token == '-LCB-':
            return '{'
        if token == '-RCB-':
            return '}'
        return token

    def tokenize(self, text):
        if False:
            return 10
        if 'NLP>' in text:
            raise RuntimeError('Bad token (NLP>) in text!')
        if text.lower().strip() == 'q':
            token = text.strip()
            index = text.index(token)
            data = [(token, text[index:], (index, index + 1), 'NN', 'q', 'O')]
            return Tokens(data, self.annotators)
        clean_text = text.replace('\n', ' ')
        self.corenlp.sendline(clean_text.encode('utf-8'))
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)
        output = self.corenlp.before
        start = output.find(b'{"sentences":')
        output = json.loads(output[start:].decode('utf-8'))
        data = []
        tokens = [t for s in output['sentences'] for t in s['tokens']]
        for i in range(len(tokens)):
            start_ws = tokens[i]['characterOffsetBegin']
            if i + 1 < len(tokens):
                end_ws = tokens[i + 1]['characterOffsetBegin']
            else:
                end_ws = tokens[i]['characterOffsetEnd']
            data.append((self._convert(tokens[i]['word']), text[start_ws:end_ws], (tokens[i]['characterOffsetBegin'], tokens[i]['characterOffsetEnd']), tokens[i].get('pos', None), tokens[i].get('lemma', None), tokens[i].get('ner', None)))
        return Tokens(data, self.annotators)