"""Utilities to measure OCR quality."""
from __future__ import annotations
import re
from collections.abc import Iterable

class OcrQualityDictionary:
    """Manages a dictionary for simple OCR quality checks."""

    def __init__(self, *, wordlist: Iterable[str]):
        if False:
            print('Hello World!')
        'Construct a dictionary from a list of words.\n\n        Words for which capitalization is important should be capitalized in the\n        dictionary. Words that contain spaces or other punctuation will never match.\n        '
        self.dictionary = set(wordlist)

    def measure_words_matched(self, ocr_text: str) -> float:
        if False:
            for i in range(10):
                print('nop')
        'Check how many unique words in the OCR text match a dictionary.\n\n        Words with mixed capitalized are only considered a match if the test word\n        matches that capitalization.\n\n        Returns:\n            number of words that match / number\n        '
        text = re.sub('[0-9_]+', ' ', ocr_text)
        text = re.sub('\\W+', ' ', text)
        text_words_list = re.split('\\s+', text)
        text_words = {w for w in text_words_list if len(w) >= 3}
        matches = 0
        for w in text_words:
            if w in self.dictionary or (w != w.lower() and w.lower() in self.dictionary):
                matches += 1
        if matches > 0:
            hit_ratio = matches / len(text_words)
        else:
            hit_ratio = 0.0
        return hit_ratio