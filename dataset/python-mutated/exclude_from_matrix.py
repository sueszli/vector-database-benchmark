from __future__ import annotations
import itertools

def representative_combos(list_1: list[str], list_2: list[str]) -> list[tuple[str, str]]:
    if False:
        i = 10
        return i + 15
    '\n    Include only representative combos from the matrix of the two lists - making sure that each of the\n    elements contributing is present at least once.\n    :param list_1: first list\n    :param list_2: second list\n    :return: list of combinations with guaranteed at least one element from each of the list\n    '
    all_selected_combinations: list[tuple[str, str]] = []
    for i in range(max(len(list_1), len(list_2))):
        all_selected_combinations.append((list_1[i % len(list_1)], list_2[i % len(list_2)]))
    return all_selected_combinations

def excluded_combos(list_1: list[str], list_2: list[str]) -> list[tuple[str, str]]:
    if False:
        i = 10
        return i + 15
    "\n    Return exclusion lists of elements that should be excluded from the matrix of the two list of items\n    if what's left should be representative list of combos (i.e. each item from both lists,\n    has to be present at least once in the combos).\n    :param list_1: first list\n    :param list_2: second list\n    :return: list of exclusions = list 1 x list 2 - representative_combos\n    "
    all_combos: list[tuple[str, str]] = list(itertools.product(list_1, list_2))
    return [item for item in all_combos if item not in set(representative_combos(list_1, list_2))]