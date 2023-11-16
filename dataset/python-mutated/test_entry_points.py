from __future__ import annotations
from typing import Iterable
from unittest import mock
from airflow.utils.entry_points import entry_points_with_dist, metadata

class MockDistribution:

    def __init__(self, name: str, entry_points: Iterable[metadata.EntryPoint]) -> None:
        if False:
            print('Hello World!')
        self.metadata = {'Name': name}
        self.entry_points = entry_points

class MockMetadata:

    def distributions(self):
        if False:
            while True:
                i = 10
        return [MockDistribution('dist1', [metadata.EntryPoint('a', 'b', 'group_x'), metadata.EntryPoint('c', 'd', 'group_y')]), MockDistribution('Dist2', [metadata.EntryPoint('e', 'f', 'group_x')]), MockDistribution('dist2', [metadata.EntryPoint('g', 'h', 'group_x')])]

@mock.patch('airflow.utils.entry_points.metadata', MockMetadata())
def test_entry_points_with_dist():
    if False:
        while True:
            i = 10
    entries = list(entry_points_with_dist('group_x'))
    assert [dist.metadata['Name'] for (_, dist) in entries] == ['dist1', 'Dist2', 'dist2']
    assert [ep.name for (ep, _) in entries] == ['a', 'e', 'g']