"""
A depparse processor which converts constituency trees using CoreNLP
"""
from stanza.pipeline._constants import TOKENIZE, CONSTITUENCY, DEPPARSE
from stanza.pipeline.processor import ProcessorVariant, register_processor_variant
from stanza.server.dependency_converter import DependencyConverter

@register_processor_variant(DEPPARSE, 'converter')
class ConverterDepparse(ProcessorVariant):
    REQUIRES_DEFAULT = set([TOKENIZE, CONSTITUENCY])

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        if config['lang'] != 'en':
            raise ValueError('Constituency to dependency converter only works for English')
        self.converter = DependencyConverter(classpath='$CLASSPATH')
        self.converter.open_pipe()

    def process(self, document):
        if False:
            while True:
                i = 10
        return self.converter.process(document)