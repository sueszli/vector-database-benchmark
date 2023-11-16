import shotchange

def test_shots_dino(capsys):
    if False:
        for i in range(10):
            print('nop')
    shotchange.analyze_shots('gs://cloud-samples-data/video/gbikes_dinosaur.mp4')
    (out, _) = capsys.readouterr()
    assert 'Shot 1:' in out