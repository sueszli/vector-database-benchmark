"""
Tests for modifications to an existing dataset.
"""
import glob
import os
import random
import shutil
import sys
import pytest

@pytest.mark.skipif(sys.version_info < (3, 10), reason='requires 3.10 glob.glob')
def test_partial_add(bench_dvc, tmp_dir, dvc, dataset, remote):
    if False:
        for i in range(10):
            print('nop')
    random.seed(4231)
    os.makedirs('partial-copy')
    for f in glob.glob('*', root_dir=dataset, recursive=True):
        if random.random() > 0.5:
            shutil.move(dataset / f, tmp_dir / 'partial-copy' / f)
    bench_dvc('add', dataset)
    bench_dvc('push')
    shutil.copytree('partial-copy', dataset, dirs_exist_ok=True)
    bench_dvc('add', dataset, name='partial')
    bench_dvc('push', name='partial')
    bench_dvc('gc', '-f', '-w', name='noop')
    bench_dvc('gc', '-f', '-w', '-c', name='cloud-noop')

@pytest.mark.skipif(sys.version_info < (3, 10), reason='requires 3.10 glob.glob')
def test_partial_remove(bench_dvc, tmp_dir, dvc, dataset, remote):
    if False:
        i = 10
        return i + 15
    random.seed(5232)
    bench_dvc('add', dataset)
    bench_dvc('push')
    for f in glob.glob('*', root_dir=dataset, recursive=True):
        if random.random() > 0.5:
            if os.path.isfile(dataset / f):
                os.remove(dataset / f)
            elif os.path.isdir(dataset / f):
                shutil.rmtree(dataset / f)
    bench_dvc('add', dataset, name='update')
    bench_dvc('push', name='update')
    bench_dvc('gc', '-f', '-w')
    bench_dvc('gc', '-f', '-w', '-c', name='cloud')