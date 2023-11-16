import os
import vision_batch_annotate_files
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

def test_sample_batch_annotate_files(capsys):
    if False:
        print('Hello World!')
    file_path = os.path.join(RESOURCES, 'kafka.pdf')
    vision_batch_annotate_files.sample_batch_annotate_files(file_path=file_path)
    (out, _) = capsys.readouterr()
    assert 'Full text' in out
    assert 'Block confidence' in out