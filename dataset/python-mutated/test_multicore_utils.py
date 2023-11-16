from metaflow.multicore_utils import parallel_map

def test_parallel_map():
    if False:
        return 10
    assert parallel_map(lambda s: s.upper(), ['a', 'b', 'c', 'd', 'e', 'f']) == ['A', 'B', 'C', 'D', 'E', 'F']