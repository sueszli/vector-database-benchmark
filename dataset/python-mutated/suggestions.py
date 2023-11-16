from __future__ import annotations
from difflib import get_close_matches
from typing import Sequence

def get_suggestion(word: str, possible_words: Sequence[str]) -> str | None:
    if False:
        print('Hello World!')
    '\n    Returns a close match of `word` amongst `possible_words`.\n\n    Args:\n        word: The word we want to find a close match for\n        possible_words: The words amongst which we want to find a close match\n\n    Returns:\n        The closest match amongst the `possible_words`. Returns `None` if no close matches could be found.\n\n    Example: returns "red" for word "redu" and possible words ("yellow", "red")\n    '
    possible_matches = get_close_matches(word, possible_words, n=1)
    return None if not possible_matches else possible_matches[0]

def get_suggestions(word: str, possible_words: Sequence[str], count: int) -> list[str]:
    if False:
        i = 10
        return i + 15
    '\n    Returns a list of up to `count` matches of `word` amongst `possible_words`.\n\n    Args:\n        word: The word we want to find a close match for\n        possible_words: The words amongst which we want to find close matches\n\n    Returns:\n        The closest matches amongst the `possible_words`, from the closest to the least close.\n            Returns an empty list if no close matches could be found.\n\n    Example: returns ["yellow", "ellow"] for word "yllow" and possible words ("yellow", "red", "ellow")\n    '
    return get_close_matches(word, possible_words, n=count)