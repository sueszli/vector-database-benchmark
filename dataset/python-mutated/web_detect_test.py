import pytest
import web_detect
ASSET_BUCKET = 'cloud-samples-data'

@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_detect_file(capsys) -> None:
    if False:
        print('Hello World!')
    file_name = '../detect/resources/landmark.jpg'
    web_detect.report(web_detect.annotate(file_name))
    (out, _) = capsys.readouterr()
    assert 'description' in out.lower()

@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_detect_web_gsuri(capsys) -> None:
    if False:
        print('Hello World!')
    file_name = 'gs://{}/vision/landmark/pofa.jpg'.format(ASSET_BUCKET)
    web_detect.report(web_detect.annotate(file_name))
    (out, _) = capsys.readouterr()
    assert 'description:' in out.lower()