from csv_to_manifest import check_errors, create_manifest_file
from os.path import exists
from os import remove
from os import path
import pytest
'\nUnit tests for csv_to_manifest.py.\n'

def clean_up(*argv):
    if False:
        while True:
            i = 10
    '\n    Deletes supplied files, if they exist.\n    Removes any previous test run files.\n    '
    for arg in argv:
        if exists(arg):
            remove(arg)

@pytest.mark.parametrize('csv_file, result', [('test/test_csvs/test_s3_supplied.csv', False)])
def test_check_no_errors(csv_file, result):
    if False:
        i = 10
        return i + 15
    '\n    Checks that the CSV file is valid.\n    '
    errors_file = f'{path.splitext(csv_file)[0]}_errors.csv'
    deduplicated_file = f'{path.splitext(csv_file)[0]}_deduplicated.csv'
    manifest_file = f'{path.splitext(csv_file)[0]}.manifest'
    clean_up(deduplicated_file, errors_file, manifest_file)
    assert check_errors(csv_file) == result
    assert not exists(deduplicated_file)
    assert not exists(errors_file)
    clean_up(deduplicated_file, errors_file, manifest_file)

@pytest.mark.parametrize('csv_file,result', [('test/test_csvs/test_dups_errors.csv', True)])
def test_check_errors(csv_file, result):
    if False:
        print('Hello World!')
    '\n    Checks that a CSV file with duplications and classification\n    errors creates the deduplication and errors CSV file.\n    '
    errors_file = f'{path.splitext(csv_file)[0]}_errors.csv'
    deduplicated_file = f'{path.splitext(csv_file)[0]}_deduplicated.csv'
    manifest_file = f'{path.splitext(csv_file)[0]}.manifest'
    clean_up(deduplicated_file, errors_file, manifest_file)
    assert check_errors(csv_file) == result
    assert exists(deduplicated_file)
    assert exists(errors_file)
    assert not exists(manifest_file)
    clean_up(deduplicated_file, errors_file, manifest_file)

@pytest.mark.parametrize('csv_file,img_count,anom_count', [('test/test_csvs/test_s3_supplied.csv', 9, 5)])
def test_create_manifest_s3_supplied(csv_file, img_count, anom_count):
    if False:
        for i in range(10):
            print('nop')
    '\n    Checks that a CSV file with images and an Amazon S3 path creates\n    a manifest file.\n    '
    s3_path = 's3://docexamplebucket1/circuitboard/train/'
    errors_file = f'{path.splitext(csv_file)[0]}_errors.csv'
    deduplicated_file = f'{path.splitext(csv_file)[0]}_deduplicated.csv'
    manifest_file = f'{path.splitext(csv_file)[0]}.manifest'
    clean_up(deduplicated_file, errors_file, manifest_file)
    (image_count, anomalous_count) = create_manifest_file(csv_file, manifest_file, s3_path)
    assert image_count == img_count
    assert anomalous_count == anom_count
    assert exists(manifest_file)
    assert not exists(deduplicated_file)
    assert not exists(errors_file)
    clean_up(deduplicated_file, errors_file, manifest_file)

@pytest.mark.parametrize('csv_file,img_count,anom_count', [('test/test_csvs/test_no_s3.csv', 7, 4)])
def test_create_manifest_no_s3_supplied(csv_file, img_count, anom_count):
    if False:
        i = 10
        return i + 15
    '\n    Checks that a CSV file with images and no Amazon S3 path creates\n    a manifest file.\n    '
    s3_path = ''
    errors_file = f'{path.splitext(csv_file)[0]}_errors.csv'
    deduplicated_file = f'{path.splitext(csv_file)[0]}_deduplicated.csv'
    manifest_file = f'{path.splitext(csv_file)[0]}.manifest'
    clean_up(deduplicated_file, errors_file, manifest_file)
    (image_count, anomalous_count) = create_manifest_file(csv_file, manifest_file, s3_path)
    assert image_count == img_count
    assert anomalous_count == anom_count
    assert exists(manifest_file)
    assert not exists(deduplicated_file)
    assert not exists(errors_file)
    clean_up(deduplicated_file, errors_file, manifest_file)