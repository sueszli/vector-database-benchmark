import labels

def test_feline_video_labels(capsys):
    if False:
        print('Hello World!')
    labels.analyze_labels('gs://cloud-samples-data/video/cat.mp4')
    (out, _) = capsys.readouterr()
    assert 'Video label description: cat' in out