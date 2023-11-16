"""
 ██████╗██╗██████╗ ██╗  ██╗███████╗██╗   ██╗
██╔════╝██║██╔══██╗██║  ██║██╔════╝╚██╗ ██╔╝
██║     ██║██████╔╝███████║█████╗   ╚████╔╝
██║     ██║██╔═══╝ ██╔══██║██╔══╝    ╚██╔╝
╚██████╗██║██║     ██║  ██║███████╗   ██║
© Brandon Skerritt
Github: brandonskerritt

Class to determine whether something is English or not.
1. Calculate the Chi Squared score of a sentence
2. If the score is significantly lower than the average score, it _might_ be English
    2.1. If the score _might_ be English, then take the text and compare it to the sorted dictionary
    in O(n log n) time.
    It creates a percentage of "How much of this text is in the dictionary?"
    The dictionary contains:
        * 20,000 most common US words
        * 10,000 most common UK words (there's no repetition between the two)
        * The top 10,000 passwords
    If the word "Looks like" English (chi-squared) and if it contains English words, we can conclude it is
    very likely English. The alternative is doing the dictionary thing but with an entire 479k word dictionary (slower)
    2.2. If the score is not English, but we haven't tested enough to create an average, then test it against
     the dictionary

Things to optimise:
* We only run the dictionary if it's 20% smaller than the average for chi squared
* We consider it "English" if 45% of the text matches the dictionary
* We run the dictionary if there is less than 10 total chisquared test

How to add a language:
* Download your desired dictionary. Try to make it the most popular words, for example. Place this file into this
 folder with languagename.txt
As an example, this comes built in with english.txt
Find the statistical frequency of each letter in that language.
For English, we have:
self.languages = {
    "English":
    [0.0855, 0.0160, 0.0316, 0.0387, 0.1210,0.0218, 0.0209, 0.0496, 0.0733, 0.0022,0.0081, 0.0421, 0.0253, 0.0717,
    0.0747,0.0207, 0.0010, 0.0633, 0.0673, 0.0894,0.0268, 0.0106, 0.0183, 0.0019, 0.0172,0.0011]
}
In chisquared.py
To add your language, do:
self.languages = {
    "English":
    [0.0855, 0.0160, 0.0316, 0.0387, 0.1210,0.0218, 0.0209, 0.0496, 0.0733, 0.0022,0.0081, 0.0421, 0.0253, 0.0717,
    0.0747,0.0207, 0.0010, 0.0633, 0.0673, 0.0894,0.0268, 0.0106, 0.0183, 0.0019, 0.0172,0.0011]
    "German": [0.0973]
}
In alphabetical order
And you're.... Done! Make sure the name of the two match up
"""
import sys
from math import ceil
from typing import Dict, Optional
import logging
from rich.logging import RichHandler
from ciphey.iface import Checker, Config, ParamSpec, T, registry
sys.path.append('..')
try:
    import mathsHelper as mh
except ModuleNotFoundError:
    import ciphey.mathsHelper as mh

@registry.register
class Brandon(Checker[str]):
    """
    Class designed to confirm whether something is **language** based on how many words of **language** appears
    Call confirmLanguage(text, language)
    * text: the text you want to confirm
    * language: the language you want to confirm

    Find out what language it is by using chisquared.py, the highest chisquared score is the language
    languageThreshold = 45
    if a string is 45% **language** words, then it's confirmed to be english
    """

    def getExpectedRuntime(self, text: T) -> float:
        if False:
            print('Hello World!')
        return 0.0001
    wordlist: set

    def clean_text(self, text: str) -> set:
        if False:
            while True:
                i = 10
        'Cleans the text ready to be checked\n\n        Strips punctuation, makes it lower case, turns it into a set separated by spaces, removes duplicate words\n\n        Args:\n            text -> The text we use to perform analysis on\n\n        Returns:\n            text -> the text as a list, now cleaned\n\n        '
        text = text.lower()
        text = self.mh.strip_punctuation(text)
        text = text.split(' ')
        text = filter(lambda x: len(x) > 2, text)
        text = set(text)
        return text

    def checker(self, text: str, threshold: float, text_length: int, var: set) -> bool:
        if False:
            return 10
        'Given text determine if it passes checker\n\n        The checker uses the variable passed to it. I.E. Stopwords list, 1k words, dictionary\n\n        Args:\n            text -> The text to check\n            threshold -> at what point do we return True? The percentage of text that is in var before we return True\n            text_length -> the length of the text\n            var -> the variable we are checking against. Stopwords list, 1k words list, dictionary list.\n        Returns:\n            boolean -> True for it passes the test, False for it fails the test.'
        if text is None:
            logging.debug("Checker's text is None, so returning False")
            return False
        if var is None:
            logging.debug("Checker's input var is None, so returning False")
            return False
        percent = ceil(text_length * threshold)
        logging.debug(f"Checker's chunks are size {percent}")
        meet_threshold = 0
        location = 0
        end = percent
        if text_length <= 0:
            return False
        while location <= text_length:
            text = list(text)
            to_analyse = text[location:end]
            logging.debug(f'To analyse is {to_analyse}')
            for word in to_analyse:
                if word in var:
                    logging.debug(f'{word} is in var, which means I am +=1 to the meet_threshold which is {meet_threshold}')
                    meet_threshold += 1
                meet_threshold_percent = meet_threshold / text_length
                if meet_threshold_percent >= threshold:
                    logging.debug(f'Returning true since the percentage is {meet_threshold / text_length} and the threshold is {threshold}')
                    return True
            location = end
            end = end + percent
        logging.debug(f'The language proportion {meet_threshold_percent} is under the threshold {threshold}')
        return False

    def __init__(self, config: Config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.mh = mh.mathsHelper()
        phases = config.get_resource(self._params()['phases'])
        self.thresholds_phase1 = phases['1']
        self.thresholds_phase2 = phases['2']
        self.top1000Words = config.get_resource(self._params().get('top1000'))
        self.wordlist = config.get_resource(self._params()['wordlist'])
        self.stopwords = config.get_resource(self._params().get('stopwords'))
        self.len_phase1 = len(self.thresholds_phase1)
        self.len_phase2 = len(self.thresholds_phase2)

    def check(self, text: str) -> Optional[str]:
        if False:
            return 10
        'Checks to see if the text is in English\n\n        Performs a decryption, but mainly parses the internal data packet and prints useful information.\n\n        Args:\n            text -> The text we use to perform analysis on\n\n        Returns:\n            bool -> True if the text is English, False otherwise.\n\n        '
        logging.debug(f'In Language Checker with "{text}"')
        text = self.clean_text(text)
        logging.debug(f'Text split to "{text}"')
        if text == '':
            logging.debug('Returning None from Brandon as the text cleaned is none.')
            return None
        length_text = len(text)
        what_to_use = {}
        what_to_use = self.calculateWhatChecker(length_text, self.thresholds_phase1.keys())
        logging.debug(self.thresholds_phase1)
        what_to_use = self.thresholds_phase1[str(what_to_use)]
        if 'check' in what_to_use:
            result = self.checker(text, what_to_use['check'], length_text, self.top1000Words)
        elif 'stop' in what_to_use:
            result = self.checker(text, what_to_use['stop'], length_text, self.stopwords)
        elif 'dict' in what_to_use:
            result = self.checker(text, what_to_use['dict'], length_text, self.wordlist)
            if not result:
                return None
        else:
            logging.info(f'It is neither stop or check, but instead {what_to_use}')
        if not result:
            return None
        else:
            what_to_use = self.calculateWhatChecker(length_text, self.thresholds_phase2.keys())
            what_to_use = self.thresholds_phase2[str(what_to_use)]
            result = self.checker(text, what_to_use['dict'], length_text, self.wordlist)
        return '' if result else None

    def calculateWhatChecker(self, length_text, key):
        if False:
            i = 10
            return i + 15
        'Calculates what threshold / checker to use\n\n        If the length of the text is over the maximum sentence length, use the last checker / threshold\n        Otherwise, traverse the keys backwards until we find a key range that does not fit.\n        So we traverse backwards and see if the sentence length is between current - 1 and current\n        In this way, we find the absolute lowest checker / percentage threshold.\n        We traverse backwards because if the text is longer than the max sentence length, we already know.\n        In total, the keys are only 5 items long or so. It is not expensive to move backwards, nor is it expensive to move forwards.\n\n        Args:\n            length_text -> The length of the text\n            key -> What key we want to use. I.E. Phase1 keys, Phase2 keys.\n        Returns:\n            what_to_use -> the key of the lowest checker.'
        _keys = list(key)
        _keys = list(map(int, _keys))
        if length_text >= int(_keys[-1]):
            what_to_use = list(key)[_keys.index(_keys[-1])]
        else:
            for (counter, i) in reversed(list(enumerate(_keys))):
                if i <= length_text:
                    what_to_use = i
        return what_to_use

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            while True:
                i = 10
        return {'top1000': ParamSpec(desc='A wordlist of the top 1000 words', req=False, default='cipheydists::list::english1000'), 'wordlist': ParamSpec(desc='A wordlist of all the words', req=False, default='cipheydists::list::english'), 'stopwords': ParamSpec(desc='A wordlist of StopWords', req=False, default='cipheydists::list::englishStopWords'), 'threshold': ParamSpec(desc='The minimum proportion (between 0 and 1) that must be in the dictionary', req=False, default=0.45), 'phases': ParamSpec(desc='Language-specific phase thresholds', req=False, default='cipheydists::brandon::english')}