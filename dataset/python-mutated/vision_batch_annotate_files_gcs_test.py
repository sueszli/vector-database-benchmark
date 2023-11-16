import os
import vision_batch_annotate_files_gcs
GCS_ROOT = 'gs://cloud-samples-data/vision/'

def test_sample_batch_annotate_files_gcs(capsys):
    if False:
        while True:
            i = 10
    storage_uri = os.path.join(GCS_ROOT, 'document_understanding/kafka.pdf')
    vision_batch_annotate_files_gcs.sample_batch_annotate_files(storage_uri=storage_uri)
    (out, _) = capsys.readouterr()
    assert 'Full text' in out
    assert 'Block confidence' in out