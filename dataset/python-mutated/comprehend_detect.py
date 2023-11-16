"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Comprehend to
detect entities, phrases, and more in a document.
"""
import logging
from pprint import pprint
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class ComprehendDetect:
    """Encapsulates Comprehend detection functions."""

    def __init__(self, comprehend_client):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param comprehend_client: A Boto3 Comprehend client.\n        '
        self.comprehend_client = comprehend_client

    def detect_languages(self, text):
        if False:
            for i in range(10):
                print('nop')
        '\n        Detects languages used in a document.\n\n        :param text: The document to inspect.\n        :return: The list of languages along with their confidence scores.\n        '
        try:
            response = self.comprehend_client.detect_dominant_language(Text=text)
            languages = response['Languages']
            logger.info('Detected %s languages.', len(languages))
        except ClientError:
            logger.exception("Couldn't detect languages.")
            raise
        else:
            return languages

    def detect_entities(self, text, language_code):
        if False:
            i = 10
            return i + 15
        '\n        Detects entities in a document. Entities can be things like people and places\n        or other common terms.\n\n        :param text: The document to inspect.\n        :param language_code: The language of the document.\n        :return: The list of entities along with their confidence scores.\n        '
        try:
            response = self.comprehend_client.detect_entities(Text=text, LanguageCode=language_code)
            entities = response['Entities']
            logger.info('Detected %s entities.', len(entities))
        except ClientError:
            logger.exception("Couldn't detect entities.")
            raise
        else:
            return entities

    def detect_key_phrases(self, text, language_code):
        if False:
            print('Hello World!')
        '\n        Detects key phrases in a document. A key phrase is typically a noun and its\n        modifiers.\n\n        :param text: The document to inspect.\n        :param language_code: The language of the document.\n        :return: The list of key phrases along with their confidence scores.\n        '
        try:
            response = self.comprehend_client.detect_key_phrases(Text=text, LanguageCode=language_code)
            phrases = response['KeyPhrases']
            logger.info('Detected %s phrases.', len(phrases))
        except ClientError:
            logger.exception("Couldn't detect phrases.")
            raise
        else:
            return phrases

    def detect_pii(self, text, language_code):
        if False:
            i = 10
            return i + 15
        '\n        Detects personally identifiable information (PII) in a document. PII can be\n        things like names, account numbers, or addresses.\n\n        :param text: The document to inspect.\n        :param language_code: The language of the document.\n        :return: The list of PII entities along with their confidence scores.\n        '
        try:
            response = self.comprehend_client.detect_pii_entities(Text=text, LanguageCode=language_code)
            entities = response['Entities']
            logger.info('Detected %s PII entities.', len(entities))
        except ClientError:
            logger.exception("Couldn't detect PII entities.")
            raise
        else:
            return entities

    def detect_sentiment(self, text, language_code):
        if False:
            while True:
                i = 10
        '\n        Detects the overall sentiment expressed in a document. Sentiment can\n        be positive, negative, neutral, or a mixture.\n\n        :param text: The document to inspect.\n        :param language_code: The language of the document.\n        :return: The sentiments along with their confidence scores.\n        '
        try:
            response = self.comprehend_client.detect_sentiment(Text=text, LanguageCode=language_code)
            logger.info('Detected primary sentiment %s.', response['Sentiment'])
        except ClientError:
            logger.exception("Couldn't detect sentiment.")
            raise
        else:
            return response

    def detect_syntax(self, text, language_code):
        if False:
            return 10
        '\n        Detects syntactical elements of a document. Syntax tokens are portions of\n        text along with their use as parts of speech, such as nouns, verbs, and\n        interjections.\n\n        :param text: The document to inspect.\n        :param language_code: The language of the document.\n        :return: The list of syntax tokens along with their confidence scores.\n        '
        try:
            response = self.comprehend_client.detect_syntax(Text=text, LanguageCode=language_code)
            tokens = response['SyntaxTokens']
            logger.info('Detected %s syntax tokens.', len(tokens))
        except ClientError:
            logger.exception("Couldn't detect syntax.")
            raise
        else:
            return tokens

def usage_demo():
    if False:
        while True:
            i = 10
    print('-' * 88)
    print('Welcome to the Amazon Comprehend detection demo!')
    print('-' * 88)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    comp_detect = ComprehendDetect(boto3.client('comprehend'))
    with open('detect_sample.txt') as sample_file:
        sample_text = sample_file.read()
    demo_size = 3
    print('Sample text used for this demo:')
    print('-' * 88)
    print(sample_text)
    print('-' * 88)
    print('Detecting languages.')
    languages = comp_detect.detect_languages(sample_text)
    pprint(languages)
    lang_code = languages[0]['LanguageCode']
    print('Detecting entities.')
    entities = comp_detect.detect_entities(sample_text, lang_code)
    print(f'The first {demo_size} are:')
    pprint(entities[:demo_size])
    print('Detecting key phrases.')
    phrases = comp_detect.detect_key_phrases(sample_text, lang_code)
    print(f'The first {demo_size} are:')
    pprint(phrases[:demo_size])
    print('Detecting personally identifiable information (PII).')
    pii_entities = comp_detect.detect_pii(sample_text, lang_code)
    print(f'The first {demo_size} are:')
    pprint(pii_entities[:demo_size])
    print('Detecting sentiment.')
    sentiment = comp_detect.detect_sentiment(sample_text, lang_code)
    print(f"Sentiment: {sentiment['Sentiment']}")
    print('SentimentScore:')
    pprint(sentiment['SentimentScore'])
    print('Detecting syntax elements.')
    syntax_tokens = comp_detect.detect_syntax(sample_text, lang_code)
    print(f'The first {demo_size} are:')
    pprint(syntax_tokens[:demo_size])
    print('Thanks for watching!')
    print('-' * 88)
if __name__ == '__main__':
    usage_demo()