"""
Tests for the :mod:`fiftyone.utils.labelbox` module.

You must run these tests interactively as follows::

    pytest tests/intensive/labelbox_tests.py -s -k <test_case>

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import unittest
from uuid import uuid4
import eta.core.utils as etau
import eta.core.web as etaw
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.labelbox as foul

def test_labelbox_image():
    if False:
        i = 10
        return i + 15
    dataset = foz.load_zoo_dataset('bdd100k', split='validation', shuffle=True, max_samples=10)
    label_field = ['weather', 'scene', 'timeofday', 'detections', 'polylines']
    _test_labelbox_image(dataset, label_field)

def test_labelbox_video_objects():
    if False:
        print('Hello World!')
    dataset = foz.load_zoo_dataset('quickstart-video', max_samples=10)
    frame_labels_field = ['detections']
    _test_labelbox_video(dataset, frame_labels_field)

def test_labelbox_video_events():
    if False:
        print('Hello World!')
    filepath = '/tmp/road.mp4'
    etaw.download_google_drive_file('1nWyKZyV6pG0hjY_gvBNShulsxLRlC6xg', path=filepath)
    dataset = fo.Dataset()
    events = [{'label': 'sunny', 'frames': [1, 10]}, {'label': 'cloudy', 'frames': [11, 20]}, {'label': 'sunny', 'frames': [21, 30]}]
    sample = fo.Sample(filepath=filepath)
    for event in events:
        label = event['label']
        frames = event['frames']
        for frame_number in range(frames[0], frames[1] + 1):
            sample.frames[frame_number]['weather'] = fo.Classification(label=label)
    dataset.add_sample(sample)
    frame_labels_field = ['weather']
    _test_labelbox_video(dataset, frame_labels_field)

def _test_labelbox_image(dataset, label_field):
    if False:
        for i in range(10):
            print('nop')
    labelbox_export_path = '/tmp/labelbox-image-export.json'
    labelbox_import_path = '/tmp/labelbox-image-import.json'
    labelbox_id_field = 'labelbox_id'
    for sample in dataset:
        sample[labelbox_id_field] = str(uuid4())
        sample.save()
    foul.export_to_labelbox(dataset, labelbox_export_path, labelbox_id_field=labelbox_id_field, label_field=label_field)
    foul.convert_labelbox_export_to_import(labelbox_export_path, labelbox_import_path)
    foul.import_from_labelbox(dataset, labelbox_import_path, label_prefix='lb', labelbox_id_field=labelbox_id_field)
    session = fo.launch_app(dataset)
    session.wait()

def _test_labelbox_video(dataset, frame_labels_field):
    if False:
        return 10
    labelbox_export_dir = '/tmp/labelbox-video-export'
    labelbox_export_path = '/tmp/labelbox-video-export.json'
    labelbox_import_dir = '/tmp/labelbox-video-import'
    labelbox_import_path = '/tmp/labelbox-video-import.json'
    labelbox_id_field = 'labelbox_id'
    etau.ensure_empty_dir(labelbox_export_dir, cleanup=True)
    etau.ensure_empty_dir(labelbox_import_dir, cleanup=True)
    for sample in dataset:
        sample[labelbox_id_field] = str(uuid4())
        sample.save()
    foul.export_to_labelbox(dataset, labelbox_export_path, video_labels_dir=labelbox_export_dir, labelbox_id_field=labelbox_id_field, frame_labels_field=frame_labels_field)
    foul.convert_labelbox_export_to_import(labelbox_export_path, outpath=labelbox_import_path, video_outdir=labelbox_import_dir)
    foul.import_from_labelbox(dataset, labelbox_import_path, label_prefix='lb', labelbox_id_field=labelbox_id_field)
    session = fo.launch_app(dataset)
    session.wait()
if __name__ == '__main__':
    fo.config.show_progress_bars = True
    unittest.main(verbosity=2)