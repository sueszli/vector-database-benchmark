import os
import re
import uuid
import backoff
from google.cloud import storage
import pytest
import detect
ASSET_BUCKET = 'cloud-samples-data'
BUCKET = os.environ['CLOUD_STORAGE_BUCKET']
OUTPUT_PREFIX = f'TEST_OUTPUT_{uuid.uuid4()}'
GCS_SOURCE_URI = f'gs://{BUCKET}/HodgeConj.pdf'
GCS_DESTINATION_URI = f'gs://{BUCKET}/{OUTPUT_PREFIX}/'

def test_labels(capsys):
    if False:
        print('Hello World!')
    file_name = os.path.join(os.path.dirname(__file__), 'resources/wakeupcat.jpg')
    detect.detect_labels(file_name)
    (out, _) = capsys.readouterr()
    assert 'Labels' in out

def test_labels_uri(capsys):
    if False:
        while True:
            i = 10
    file_name = f'gs://{ASSET_BUCKET}/vision/label/wakeupcat.jpg'
    detect.detect_labels_uri(file_name)
    (out, _) = capsys.readouterr()
    assert 'Labels' in out

def test_landmarks(capsys):
    if False:
        return 10
    file_name = os.path.join(os.path.dirname(__file__), 'resources/landmark.jpg')
    detect.detect_landmarks(file_name)
    (out, _) = capsys.readouterr()
    assert 'palace' in out.lower()

def test_landmarks_uri(capsys):
    if False:
        while True:
            i = 10
    file_name = f'gs://{ASSET_BUCKET}/vision/landmark/pofa.jpg'
    detect.detect_landmarks_uri(file_name)
    (out, _) = capsys.readouterr()
    assert 'palace' in out.lower()

def test_faces(capsys):
    if False:
        return 10
    file_name = os.path.join(os.path.dirname(__file__), 'resources/face_no_surprise.jpg')
    detect.detect_faces(file_name)
    (out, _) = capsys.readouterr()
    assert 'face bound' in out

def test_faces_uri(capsys):
    if False:
        print('Hello World!')
    file_name = f'gs://{ASSET_BUCKET}/vision/face/face_no_surprise.jpg'
    detect.detect_faces_uri(file_name)
    (out, _) = capsys.readouterr()
    assert 'face bound' in out

def test_logos(capsys):
    if False:
        for i in range(10):
            print('nop')
    file_name = os.path.join(os.path.dirname(__file__), 'resources/logos.png')
    detect.detect_logos(file_name)
    (out, _) = capsys.readouterr()
    assert 'google' in out.lower()

def test_logos_uri(capsys):
    if False:
        for i in range(10):
            print('nop')
    file_name = f'gs://{ASSET_BUCKET}/vision/logo/logo_google.png'
    detect.detect_logos_uri(file_name)
    (out, _) = capsys.readouterr()
    assert 'google' in out.lower()

def test_safe_search(capsys):
    if False:
        return 10
    file_name = os.path.join(os.path.dirname(__file__), 'resources/wakeupcat.jpg')
    detect.detect_safe_search(file_name)
    (out, _) = capsys.readouterr()
    assert 'VERY_LIKELY' in out
    assert 'racy: ' in out

def test_safe_search_uri(capsys):
    if False:
        for i in range(10):
            print('nop')
    file_name = f'gs://{ASSET_BUCKET}/vision/label/wakeupcat.jpg'
    detect.detect_safe_search_uri(file_name)
    (out, _) = capsys.readouterr()
    assert 'VERY_LIKELY' in out
    assert 'racy: ' in out

def test_detect_text(capsys):
    if False:
        i = 10
        return i + 15
    file_name = os.path.join(os.path.dirname(__file__), 'resources/text.jpg')
    detect.detect_text(file_name)
    (out, _) = capsys.readouterr()
    assert '37%' in out

def test_detect_text_uri(capsys):
    if False:
        for i in range(10):
            print('nop')
    file_name = f'gs://{ASSET_BUCKET}/vision/text/screen.jpg'
    detect.detect_text_uri(file_name)
    (out, _) = capsys.readouterr()
    assert '37%' in out

def test_detect_properties(capsys):
    if False:
        i = 10
        return i + 15
    file_name = os.path.join(os.path.dirname(__file__), 'resources/landmark.jpg')
    detect.detect_properties(file_name)
    (out, _) = capsys.readouterr()
    assert 'frac' in out

def test_detect_properties_uri(capsys):
    if False:
        while True:
            i = 10
    file_name = f'gs://{ASSET_BUCKET}/vision/landmark/pofa.jpg'
    detect.detect_properties_uri(file_name)
    (out, _) = capsys.readouterr()
    assert 'frac' in out

def only_sample_error(e):
    if False:
        print('Hello World!')
    'A callback for giving up upon Exceptions.\n\n    Giving up upon any Exceptions other than the ones that sample code\n    throws at the end of the function.\n    '
    return 'https://cloud.google.com/apis/design/errors' not in str(e)

def test_detect_web(capsys):
    if False:
        print('Hello World!')
    file_name = os.path.join(os.path.dirname(__file__), 'resources/landmark.jpg')

    @backoff.on_exception(backoff.expo, Exception, max_time=60, giveup=only_sample_error)
    def run_sample():
        if False:
            return 10
        detect.detect_web(file_name)
    run_sample()
    (out, _) = capsys.readouterr()
    assert re.search('best guess label:.*palace of fine arts', out, re.DOTALL | re.I)

def test_detect_web_uri(capsys):
    if False:
        while True:
            i = 10
    file_name = f'gs://{ASSET_BUCKET}/vision/landmark/pofa.jpg'

    @backoff.on_exception(backoff.expo, Exception, max_time=60, giveup=only_sample_error)
    def run_sample():
        if False:
            return 10
        detect.detect_web_uri(file_name)
    run_sample()
    (out, _) = capsys.readouterr()
    assert re.search('best guess label:.*palace of fine arts', out, re.DOTALL | re.I)

def test_detect_web_with_geo(capsys):
    if False:
        i = 10
        return i + 15
    file_name = os.path.join(os.path.dirname(__file__), 'resources/city.jpg')

    @backoff.on_exception(backoff.expo, Exception, max_time=60, giveup=only_sample_error)
    def run_sample():
        if False:
            print('Hello World!')
        detect.web_entities_include_geo_results(file_name)
    run_sample()
    (out, _) = capsys.readouterr()
    out = out.lower()
    assert 'description' in out

def test_detect_web_with_geo_uri(capsys):
    if False:
        i = 10
        return i + 15
    file_name = f'gs://{ASSET_BUCKET}/vision/web/city.jpg'

    @backoff.on_exception(backoff.expo, Exception, max_time=60, giveup=only_sample_error)
    def run_sample():
        if False:
            while True:
                i = 10
        detect.web_entities_include_geo_results_uri(file_name)
    run_sample()
    (out, _) = capsys.readouterr()
    out = out.lower()
    assert 'description' in out

def test_detect_document(capsys):
    if False:
        for i in range(10):
            print('nop')
    file_name = os.path.join(os.path.dirname(__file__), 'resources/text.jpg')
    detect.detect_document(file_name)
    (out, _) = capsys.readouterr()
    assert 'class' in out

def test_detect_document_uri(capsys):
    if False:
        i = 10
        return i + 15
    file_name = f'gs://{ASSET_BUCKET}/vision/text/screen.jpg'
    detect.detect_document_uri(file_name)
    (out, _) = capsys.readouterr()
    assert 'class' in out

def test_detect_crop_hints(capsys):
    if False:
        print('Hello World!')
    file_name = os.path.join(os.path.dirname(__file__), 'resources/wakeupcat.jpg')
    detect.detect_crop_hints(file_name)
    (out, _) = capsys.readouterr()
    assert 'bounds: ' in out

def test_detect_crop_hints_uri(capsys):
    if False:
        i = 10
        return i + 15
    file_name = f'gs://{ASSET_BUCKET}/vision/label/wakeupcat.jpg'
    detect.detect_crop_hints_uri(file_name)
    (out, _) = capsys.readouterr()
    assert 'bounds: ' in out

@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_async_detect_document(capsys):
    if False:
        for i in range(10):
            print('nop')
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET)
    if len(list(bucket.list_blobs(prefix=OUTPUT_PREFIX))) > 0:
        for blob in bucket.list_blobs(prefix=OUTPUT_PREFIX):
            blob.delete()
    assert len(list(bucket.list_blobs(prefix=OUTPUT_PREFIX))) == 0
    uri = 'gs://{}/vision/document/custom_0773375000_title_only.pdf'.format(ASSET_BUCKET)
    detect.async_detect_document(gcs_source_uri=uri, gcs_destination_uri=GCS_DESTINATION_URI)
    (out, _) = capsys.readouterr()
    assert 'OIL, GAS AND MINERAL LEASE' in out
    assert len(list(bucket.list_blobs(prefix=OUTPUT_PREFIX))) > 0
    for blob in bucket.list_blobs(prefix=OUTPUT_PREFIX):
        blob.delete()
    assert len(list(bucket.list_blobs(prefix=OUTPUT_PREFIX))) == 0

def test_localize_objects(capsys):
    if False:
        while True:
            i = 10
    detect.localize_objects('resources/puppies.jpg')
    (out, _) = capsys.readouterr()
    assert 'dog' in out.lower()

def test_localize_objects_uri(capsys):
    if False:
        i = 10
        return i + 15
    uri = 'gs://cloud-samples-data/vision/puppies.jpg'
    detect.localize_objects_uri(uri)
    (out, _) = capsys.readouterr()
    assert 'dog' in out.lower()