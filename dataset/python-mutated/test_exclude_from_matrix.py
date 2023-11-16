from __future__ import annotations
import pytest
from airflow_breeze.utils.exclude_from_matrix import excluded_combos, representative_combos

@pytest.mark.parametrize('list_1, list_2, expected_representative_list', [(['3.8', '3.9'], ['1', '2'], [('3.8', '1'), ('3.9', '2')]), (['3.8', '3.9'], ['1', '2', '3'], [('3.8', '1'), ('3.9', '2'), ('3.8', '3')]), (['3.8', '3.9'], ['1', '2', '3', '4'], [('3.8', '1'), ('3.9', '2'), ('3.8', '3'), ('3.9', '4')]), (['3.8', '3.9', '3.10'], ['1', '2', '3', '4'], [('3.8', '1'), ('3.9', '2'), ('3.10', '3'), ('3.8', '4')])])
def test_exclude_from_matrix(list_1: list[str], list_2: list[str], expected_representative_list: dict[str, str]):
    if False:
        for i in range(10):
            print('nop')
    representative_list = representative_combos(list_1, list_2)
    exclusion_list = excluded_combos(list_1, list_2)
    assert representative_list == expected_representative_list
    assert len(representative_list) == len(list_1) * len(list_2) - len(exclusion_list)
    assert not set(representative_list).intersection(exclusion_list)