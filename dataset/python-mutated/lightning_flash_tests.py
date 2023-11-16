"""
Tests for the :mod:`fiftyone.utils.flash` module.

You must run these tests interactively as follows::

    python tests/intensive/lightning_flash_tests.py

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import numpy as np
import unittest
from itertools import chain
from flash.core.data.utils import download_data
from flash.image.detection.output import FiftyOneDetectionLabelsOutput
from flash.core.classification import FiftyOneLabelsOutput
from flash.image.segmentation.output import FiftyOneSegmentationLabelsOutput
from flash.image import ImageClassificationData, ImageClassifier, ImageEmbedder
from flash.image import ObjectDetectionData, ObjectDetector
from flash.image import SemanticSegmentation, SemanticSegmentationData
from flash import Trainer
from flash.video import VideoClassificationData, VideoClassifier
import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.utils.random as four
import fiftyone.zoo as foz

class LightningFlashTests(unittest.TestCase):

    def test_apply_model(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = foz.load_zoo_dataset('quickstart', max_samples=5).clone()
        num_classes = len(dataset.distinct('ground_truth.detections.label'))
        cls_model = ImageClassifier(backbone='resnet18', num_classes=num_classes)
        det_model = ObjectDetector(head='efficientdet', backbone='d0', num_classes=num_classes, image_size=512)
        dataset.apply_model(cls_model, label_field='flash_classifications')
        transform_kwargs = {'image_size': 512}
        dataset.apply_model(det_model, label_field='flash_detections', transform_kwargs=transform_kwargs)

    def test_image_classifier(self):
        if False:
            while True:
                i = 10
        dataset = foz.load_zoo_dataset('cifar10', split='test', max_samples=300).clone()
        dataset.untag_samples('test')
        splits = {'train': 0.7, 'test': 0.1, 'val': 0.1, 'pred': 0.1}
        four.random_split(dataset, splits)
        train_dataset = dataset.match_tags('train')
        test_dataset = dataset.match_tags('test')
        val_dataset = dataset.match_tags('val')
        predict_dataset = dataset.match_tags('pred')
        datamodule = ImageClassificationData.from_fiftyone(train_dataset=train_dataset, test_dataset=test_dataset, val_dataset=val_dataset, predict_dataset=predict_dataset, label_field='ground_truth', batch_size=4, num_workers=4)
        model = ImageClassifier(backbone='resnet18', labels=datamodule.labels)
        trainer = Trainer(max_epochs=1, limit_train_batches=10, limit_val_batches=10)
        trainer.finetune(model, datamodule=datamodule)
        trainer.save_checkpoint('/tmp/image_classification_model.pt')
        predictions = trainer.predict(model, datamodule=datamodule, output=FiftyOneLabelsOutput(labels=datamodule.labels))
        predictions = list(chain.from_iterable(predictions))
        predictions = {p['filepath']: p['predictions'] for p in predictions}
        predict_dataset.set_values('flash_predictions', predictions, key_field='filepath')

    def test_object_detector(self):
        if False:
            i = 10
            return i + 15
        dataset = foz.load_zoo_dataset('coco-2017', split='validation', max_samples=100, classes=['person']).clone()
        splits = {'train': 0.7, 'test': 0.1, 'val': 0.1}
        four.random_split(dataset, splits)
        train_dataset = dataset.match_tags('train')
        test_dataset = dataset.match_tags('test')
        val_dataset = dataset.match_tags('val')
        predict_dataset = train_dataset.take(5)
        dataset.default_classes.pop(0)
        datamodule = ObjectDetectionData.from_fiftyone(train_dataset=train_dataset, test_dataset=test_dataset, val_dataset=val_dataset, predict_dataset=predict_dataset, label_field='ground_truth', transform_kwargs={'image_size': 512}, batch_size=4)
        model = ObjectDetector(head='efficientdet', backbone='d0', num_classes=datamodule.num_classes, image_size=512)
        trainer = Trainer(max_epochs=1, limit_train_batches=10)
        trainer.finetune(model, datamodule=datamodule, strategy='freeze')
        trainer.save_checkpoint('/tmp/object_detection_model.pt')
        predictions = trainer.predict(model, datamodule=datamodule, output=FiftyOneDetectionLabelsOutput(labels=datamodule.labels))
        predictions = list(chain.from_iterable(predictions))
        predictions = {p['filepath']: p['predictions'] for p in predictions}
        dataset.set_values('flash_predictions', predictions, key_field='filepath')

    def test_semantic_segmentation(self):
        if False:
            return 10
        download_data('https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip', '/tmp/carla_data/')
        dataset = fo.Dataset.from_dir(dataset_dir='/tmp/carla_data', dataset_type=fo.types.ImageSegmentationDirectory, data_path='CameraRGB', labels_path='CameraSeg', force_grayscale=True, shuffle=True)
        predict_dataset = dataset.take(5)
        datamodule = SemanticSegmentationData.from_fiftyone(train_dataset=dataset, test_dataset=dataset, val_dataset=dataset, predict_dataset=predict_dataset, label_field='ground_truth', transform_kwargs=dict(image_size=(256, 256)), num_classes=21, batch_size=4)
        model = SemanticSegmentation(backbone='mobilenetv3_large_100', head='fpn', num_classes=datamodule.num_classes)
        trainer = Trainer(max_epochs=1, limit_train_batches=10, limit_val_batches=5)
        trainer.finetune(model, datamodule=datamodule, strategy='freeze')
        trainer.save_checkpoint('/tmp/semantic_segmentation_model.pt')
        predictions = trainer.predict(model, datamodule=datamodule, output=FiftyOneSegmentationLabelsOutput())
        predictions = list(chain.from_iterable(predictions))
        predictions = {p['filepath']: p['predictions'] for p in predictions}
        dataset.set_values('flash_predictions', predictions, key_field='filepath')
        predict_dataset.apply_model(model, 'seg_apply_model')

    def test_video_classification(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = foz.load_zoo_dataset('kinetics-700-2020', split='validation', max_samples=15, shuffle=True).clone()
        dataset.untag_samples('validation')
        labels = dataset.distinct('ground_truth.label')
        labels_map = {l: l.replace(' ', '_') for l in labels}
        dataset = dataset.map_labels('ground_truth', labels_map).clone()
        labels = dataset.distinct('ground_truth.label')
        splits = {'train': 0.7, 'pred': 0.3}
        four.random_split(dataset, splits)
        train_dataset = dataset.match_tags('train')
        predict_dataset = dataset.match_tags('pred')
        datamodule = VideoClassificationData.from_fiftyone(train_dataset=dataset, predict_dataset=predict_dataset, label_field='ground_truth', batch_size=1, clip_sampler='uniform', clip_duration=1, decode_audio=False)
        model = VideoClassifier(backbone='x3d_xs', labels=datamodule.labels, pretrained=False)
        trainer = Trainer(max_epochs=1, limit_train_batches=5)
        trainer.finetune(model, datamodule=datamodule, strategy='freeze')
        trainer.save_checkpoint('/tmp/video_classification.pt')
        predictions = trainer.predict(model, datamodule=datamodule, output=FiftyOneLabelsOutput(labels=datamodule.labels))
        predictions = list(chain.from_iterable(predictions))
        predictions = {p['filepath']: p['predictions'] for p in predictions}
        predict_dataset.set_values('flash_predictions', predictions, key_field='filepath')

    def test_manually_adding_predictions(self):
        if False:
            i = 10
            return i + 15
        dataset = foz.load_zoo_dataset('quickstart', max_samples=5).select_fields('ground_truth').clone()
        labels = dataset.distinct('ground_truth.detections.label')
        model = ImageClassifier(labels=labels)
        datamodule = ImageClassificationData.from_fiftyone(predict_dataset=dataset, batch_size=1)
        output = FiftyOneLabelsOutput(return_filepath=False, labels=labels)
        predictions = Trainer().predict(model, datamodule=datamodule, output=output)
        predictions = list(chain.from_iterable(predictions))
        dataset.set_values('flash_predictions', predictions)

    def test_specifying_class_names(self):
        if False:
            return 10
        dataset = foz.load_zoo_dataset('quickstart', max_samples=5).clone()
        datamodule = ImageClassificationData.from_fiftyone(predict_dataset=dataset, batch_size=1)
        num_classes = 100
        model = ImageClassifier(backbone='resnet18', num_classes=num_classes)
        labels = ['label_' + str(i) for i in range(num_classes)]
        output = FiftyOneLabelsOutput(labels=labels)
        trainer = Trainer()
        predictions = trainer.predict(model, datamodule=datamodule, output=output)
        predictions = list(chain.from_iterable(predictions))
        predictions = {p['filepath']: p['predictions'] for p in predictions}
        dataset.set_values('flash_predictions', predictions, key_field='filepath')
        print(dataset.distinct('flash_predictions.label'))

    def test_image_embedder(self):
        if False:
            print('Hello World!')
        download_data('https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip', '/tmp')
        dataset = fo.Dataset.from_dir('/tmp/hymenoptera_data/test/', fo.types.ImageClassificationDirectoryTree)
        datamodule = ImageClassificationData.from_fiftyone(predict_dataset=dataset, batch_size=1)
        embedder = ImageEmbedder(backbone='vision_transformer', training_strategy='barlow_twins', head='barlow_twins_head', pretraining_transform='barlow_twins_transform', training_strategy_kwargs={'latent_embedding_dim': 128}, pretraining_transform_kwargs={'size_crops': [32]})
        trainer = Trainer()
        embedding_batches = trainer.predict(embedder, datamodule=datamodule)
        embeddings = np.stack(sum(embedding_batches, []))
        results = fob.compute_visualization(dataset, embeddings=embeddings)
        plot = results.visualize(labels='ground_truth.label')
        plot.show()
        embeddings = dataset.compute_embeddings(embedder)
        dataset.compute_embeddings(embedder, embeddings_field='embeddings')
if __name__ == '__main__':
    fo.config.show_progress_bars = False
    unittest.main(verbosity=2)