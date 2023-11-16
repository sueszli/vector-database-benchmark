from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import pytest
import tensorflow as tf
from tests.utils import ARTTestException, master_seed
logger = logging.getLogger(__name__)

@pytest.mark.skip_module('object_detection')
@pytest.mark.skip_framework('pytorch', 'tensorflow2v1', 'keras', 'kerastf', 'mxnet', 'non_dl_frameworks')
def test_tf_faster_rcnn(art_warning, get_mnist_dataset):
    if False:
        print('Hello World!')
    try:
        master_seed(seed=1234, set_tensorflow=True)
        from art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
        images = tf.placeholder(tf.float32, shape=[1, 28, 28, 1])
        obj_dec = TensorFlowFasterRCNN(images=images)
        ((_, _), (x_test_mnist, y_test_mnist)) = get_mnist_dataset
        x_test_mnist = x_test_mnist[:1]
        result = obj_dec.predict(x_test_mnist)
        assert list(result[0].keys()) == ['boxes', 'labels', 'scores']
        assert result[0]['boxes'].shape == (300, 4)
        expected_detection_boxes = np.asarray([0.008862, 0.003788, 0.070454, 0.175931])
        np.testing.assert_array_almost_equal(result[0]['boxes'][2, :], expected_detection_boxes, decimal=3)
        assert result[0]['scores'].shape == (300,)
        expected_detection_scores = np.asarray([0.0002196349, 7.968055e-05, 7.811916e-05, 7.334248e-05, 6.868376e-05, 6.861838e-05, 6.756858e-05, 6.331169e-05, 6.313509e-05, 6.222352e-05])
        np.testing.assert_array_almost_equal(result[0]['scores'][:10], expected_detection_scores, decimal=3)
        assert result[0]['labels'].shape == (300,)
        y = [{'boxes': result[0]['boxes'], 'labels': result[0]['labels'], 'scores': np.ones_like(result[0]['labels'])}]
        grads = obj_dec.loss_gradient(x_test_mnist[:1], y)
        assert grads.shape == (1, 28, 28, 1)
        expected_gradients = np.asarray([[-0.00298723], [-0.0039893], [-0.00036253], [0.01038542], [0.01455704], [0.00995643], [0.00424966], [0.00470569], [0.00666382], [0.0028694], [0.00525351], [0.00889174], [0.0071413], [0.00618231], [0.00598106], [0.0072665], [0.00708815], [0.00286943], [0.00411595], [0.00788978], [0.00587319], [0.00808631], [0.01018151], [0.00867905], [0.00820272], [0.00124911], [-0.0042593], [0.02380728]])
        np.testing.assert_array_almost_equal(grads[0, 0, :, :], expected_gradients, decimal=2)
        result_tf = obj_dec.predict(x_test_mnist, standardise_output=False)
        result = obj_dec.predict(x_test_mnist, standardise_output=True)
        from art.estimators.object_detection.utils import convert_tf_to_pt
        result_pt = convert_tf_to_pt(y=result_tf, height=x_test_mnist.shape[1], width=x_test_mnist.shape[2])
        np.testing.assert_array_equal(result[0]['boxes'], result_pt[0]['boxes'])
        np.testing.assert_array_equal(result[0]['labels'], result_pt[0]['labels'])
        np.testing.assert_array_equal(result[0]['scores'], result_pt[0]['scores'])
        y = [{'boxes': result[0]['boxes'], 'labels': result[0]['labels'], 'scores': np.ones_like(result[0]['labels'])}]
        grads = obj_dec.loss_gradient(x_test_mnist[:1], y, standardise_output=True)
        assert grads.shape == (1, 28, 28, 1)
        expected_gradients = np.asarray([[-0.00095965], [-0.00265362], [-0.00031886], [0.01132964], [0.01674244], [0.01262039], [0.0063345], [0.00673249], [0.00618648], [0.00422678], [0.00542425], [0.00814896], [0.00919153], [0.01068758], [0.00929435], [0.00877143], [0.00747379], [0.0050377], [0.00656254], [0.00799547], [0.0051057], [0.00714598], [0.01090685], [0.00787637], [0.00709959], [0.00047201], [-0.00460457], [0.02629307]])
        np.testing.assert_array_almost_equal(grads[0, 0, :, :], expected_gradients, decimal=2)
        obj_dec._sess.close()
        tf.reset_default_graph()
    except ARTTestException as e:
        art_warning(e)

def test_errors(art_warning, get_mnist_dataset):
    if False:
        while True:
            i = 10
    try:
        from art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
        images = tf.placeholder(tf.float32, shape=[1, 28, 28, 1])
        with pytest.raises(ValueError):
            TensorFlowFasterRCNN(images=images, clip_values=(1, 2), attack_losses=('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'))
        with pytest.raises(ValueError):
            TensorFlowFasterRCNN(images=images, clip_values=(-1, 1), attack_losses=('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'))
        from art.defences.postprocessor.rounded import Rounded
        post_def = Rounded()
        with pytest.raises(ValueError):
            TensorFlowFasterRCNN(images=images, clip_values=(0, 1), attack_losses=('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'), postprocessing_defences=post_def)
    except ARTTestException as e:
        art_warning(e)

def test_preprocessing_defences(art_warning, get_mnist_dataset):
    if False:
        print('Hello World!')
    try:
        from art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
        from art.defences.preprocessor.spatial_smoothing import SpatialSmoothing
        images = tf.placeholder(tf.float32, shape=[1, 28, 28, 3])
        pre_def = SpatialSmoothing()
        with pytest.raises(ValueError):
            _ = TensorFlowFasterRCNN(images=images, clip_values=(0, 1), attack_losses=('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'), preprocessing_defences=pre_def)
    except ARTTestException as e:
        art_warning(e)

def test_compute_losses(art_warning, get_mnist_dataset):
    if False:
        for i in range(10):
            print('nop')
    try:
        from art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
        images = tf.placeholder(tf.float32, shape=[2, 28, 28, 3])
        ((_, _), (x_test_mnist, y_test_mnist)) = get_mnist_dataset
        frcnn = TensorFlowFasterRCNN(images=images, clip_values=(0, 1), attack_losses=('Loss/RPNLoss/localization_loss', 'Loss/RPNLoss/objectness_loss', 'Loss/BoxClassifierLoss/localization_loss', 'Loss/BoxClassifierLoss/classification_loss'))
        result = frcnn.predict(np.repeat(x_test_mnist[:2].astype(np.float32), repeats=3, axis=3))
        y = [{'boxes': result[0]['boxes'], 'labels': result[0]['labels'], 'scores': np.ones_like(result[0]['labels'])}, {'boxes': result[1]['boxes'], 'labels': result[1]['labels'], 'scores': np.ones_like(result[1]['labels'])}]
        losses = frcnn.compute_losses(np.repeat(x_test_mnist[:2].astype(np.float32), repeats=3, axis=3), y)
        assert len(losses) == 4
    except ARTTestException as e:
        art_warning(e)

def test_compute_loss(art_warning, get_mnist_dataset):
    if False:
        return 10
    try:
        from art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
        images = tf.placeholder(tf.float32, shape=[2, 28, 28, 3])
        ((_, _), (x_test_mnist, y_test_mnist)) = get_mnist_dataset
        frcnn = TensorFlowFasterRCNN(images=images, clip_values=(0, 1), attack_losses=('Loss/RPNLoss/localization_loss', 'Loss/RPNLoss/objectness_loss', 'Loss/BoxClassifierLoss/localization_loss', 'Loss/BoxClassifierLoss/classification_loss'))
        result = frcnn.predict(np.repeat(x_test_mnist[:2].astype(np.float32), repeats=3, axis=3))
        y = [{'boxes': result[0]['boxes'], 'labels': result[0]['labels'], 'scores': np.ones_like(result[0]['labels'])}, {'boxes': result[1]['boxes'], 'labels': result[1]['labels'], 'scores': np.ones_like(result[1]['labels'])}]
        loss = frcnn.compute_loss(np.repeat(x_test_mnist[:2].astype(np.float32), repeats=3, axis=3), y)
        assert float(loss) == pytest.approx(6.592308044433594, abs=0.5)
    except ARTTestException as e:
        art_warning(e)