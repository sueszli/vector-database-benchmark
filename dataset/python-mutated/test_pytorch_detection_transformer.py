from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import pytest
logger = logging.getLogger(__name__)

@pytest.fixture()
@pytest.mark.skip_framework('tensorflow', 'tensorflow2v1', 'keras', 'kerastf', 'mxnet', 'non_dl_frameworks')
def get_pytorch_detr():
    if False:
        print('Hello World!')
    from art.utils import load_dataset
    from art.estimators.object_detection.pytorch_detection_transformer import PyTorchDetectionTransformer
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    INPUT_SHAPE = (3, 32, 32)
    object_detector = PyTorchDetectionTransformer(input_shape=INPUT_SHAPE, clip_values=(0, 1), preprocessing=(MEAN, STD))
    n_test = 2
    ((_, _), (x_test, y_test), _, _) = load_dataset('cifar10')
    x_test = x_test.transpose(0, 3, 1, 2).astype(np.float32)
    x_test = x_test[:n_test]
    result = object_detector.predict(x=x_test)
    y_test = [{'boxes': result[0]['boxes'], 'labels': result[0]['labels'], 'scores': np.ones_like(result[0]['labels'])}, {'boxes': result[1]['boxes'], 'labels': result[1]['labels'], 'scores': np.ones_like(result[1]['labels'])}]
    yield (object_detector, x_test, y_test)

@pytest.mark.only_with_platform('pytorch')
def test_predict(get_pytorch_detr):
    if False:
        return 10
    (object_detector, x_test, _) = get_pytorch_detr
    result = object_detector.predict(x=x_test)
    assert list(result[0].keys()) == ['boxes', 'labels', 'scores']
    assert result[0]['boxes'].shape == (100, 4)
    expected_detection_boxes = np.asarray([-0.0059490204, 11.947733, 31.993944, 31.925127])
    np.testing.assert_array_almost_equal(result[0]['boxes'][2, :], expected_detection_boxes, decimal=1)
    assert result[0]['scores'].shape == (100,)
    expected_detection_scores = np.asarray([0.00679839, 0.0250559, 0.07205943, 0.01115368, 0.03321039, 0.10407761, 0.00113309, 0.01442852, 0.00527624, 0.01240906])
    np.testing.assert_array_almost_equal(result[0]['scores'][:10], expected_detection_scores, decimal=1)
    assert result[0]['labels'].shape == (100,)
    expected_detection_classes = np.asarray([17, 17, 33, 17, 17, 17, 74, 17, 17, 17])
    np.testing.assert_array_almost_equal(result[0]['labels'][:10], expected_detection_classes, decimal=5)

@pytest.mark.only_with_platform('pytorch')
def test_loss_gradient(get_pytorch_detr):
    if False:
        while True:
            i = 10
    (object_detector, x_test, y_test) = get_pytorch_detr
    grads = object_detector.loss_gradient(x=x_test, y=y_test)
    assert grads.shape == (2, 3, 800, 800)
    expected_gradients1 = np.asarray([-0.00061366, 0.00322502, -0.00039866, -0.00807413, -0.00476555, 0.00181204, 0.01007765, 0.00415828, -0.00073114, 0.00018387, -0.00146992, -0.00119636, -0.00098966, -0.00295517, -0.0024271, -0.00131314, -0.00149217, -0.00104926, -0.00154239, -0.00110989, 0.00092887, 0.00049146, -0.00292508, -0.00124526, 0.00140347, 0.00019833, 0.00191074, -0.00117537, -0.00080604, 0.00057427, -0.00061728, -0.00206535])
    np.testing.assert_array_almost_equal(grads[0, 0, 10, :32], expected_gradients1, decimal=2)
    expected_gradients2 = np.asarray([-0.001178753, -0.002850068, 0.005088497, 0.00064504531, -6.8841036e-05, 0.0028184296, 0.0030257765, 0.00028565727, -0.00010701057, 0.0012945699, 0.00073593057, 0.0010177144, -0.0024692707, -0.0013801848, 0.0006318228, -0.00042305476, 0.0004430775, 0.00085821096, -0.00071204413, -0.0031404425, -0.0015964351, -0.0019222996, -0.00053157361, -0.00099202688, -0.0015815455, 0.00020060266, -0.0020584739, 0.00066960667, 0.00097393827, -0.0016040013, -0.00069741381, 0.00014657658])
    np.testing.assert_array_almost_equal(grads[1, 0, 10, :32], expected_gradients2, decimal=2)

@pytest.mark.only_with_platform('pytorch')
def test_errors():
    if False:
        for i in range(10):
            print('nop')
    from torch import hub
    from art.estimators.object_detection.pytorch_detection_transformer import PyTorchDetectionTransformer
    model = hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    with pytest.raises(ValueError):
        PyTorchDetectionTransformer(model=model, clip_values=(1, 2), attack_losses=('loss_ce', 'loss_bbox', 'loss_giou'))
    with pytest.raises(ValueError):
        PyTorchDetectionTransformer(model=model, clip_values=(-1, 1), attack_losses=('loss_ce', 'loss_bbox', 'loss_giou'))
    from art.defences.postprocessor.rounded import Rounded
    post_def = Rounded()
    with pytest.raises(ValueError):
        PyTorchDetectionTransformer(model=model, clip_values=(0, 1), attack_losses=('loss_ce', 'loss_bbox', 'loss_giou'), postprocessing_defences=post_def)

@pytest.mark.only_with_platform('pytorch')
def test_preprocessing_defences(get_pytorch_detr):
    if False:
        print('Hello World!')
    (object_detector, x_test, _) = get_pytorch_detr
    from art.defences.preprocessor.spatial_smoothing_pytorch import SpatialSmoothingPyTorch
    pre_def = SpatialSmoothingPyTorch()
    object_detector.set_params(preprocessing_defences=pre_def)
    result = object_detector.predict(x=x_test)
    y = [{'boxes': result[0]['boxes'], 'labels': result[0]['labels'], 'scores': np.ones_like(result[0]['labels'])}, {'boxes': result[1]['boxes'], 'labels': result[1]['labels'], 'scores': np.ones_like(result[1]['labels'])}]
    grads = object_detector.loss_gradient(x=x_test, y=y)
    assert grads.shape == (2, 3, 800, 800)

@pytest.mark.only_with_platform('pytorch')
def test_compute_losses(get_pytorch_detr):
    if False:
        i = 10
        return i + 15
    (object_detector, x_test, y_test) = get_pytorch_detr
    object_detector.attack_losses = 'loss_ce'
    losses = object_detector.compute_losses(x=x_test, y=y_test)
    assert len(losses) == 1

@pytest.mark.only_with_platform('pytorch')
def test_compute_loss(get_pytorch_detr):
    if False:
        while True:
            i = 10
    (object_detector, x_test, _) = get_pytorch_detr
    result = object_detector.predict(x_test)
    y = [{'boxes': result[0]['boxes'], 'labels': result[0]['labels'], 'scores': np.ones_like(result[0]['labels'])}, {'boxes': result[1]['boxes'], 'labels': result[1]['labels'], 'scores': np.ones_like(result[1]['labels'])}]
    loss = object_detector.compute_loss(x=x_test, y=y)
    assert pytest.approx(3.9634, abs=0.01) == float(loss)

@pytest.mark.only_with_platform('pytorch')
def test_pgd(get_pytorch_detr):
    if False:
        for i in range(10):
            print('nop')
    (object_detector, x_test, y_test) = get_pytorch_detr
    from art.attacks.evasion import ProjectedGradientDescent
    from PIL import Image
    imgs = []
    for i in x_test:
        img = Image.fromarray((i * 255).astype(np.uint8).transpose(1, 2, 0))
        img = img.resize(size=(800, 800))
        imgs.append(np.array(img))
    x_test = np.array(imgs).transpose(0, 3, 1, 2)
    attack = ProjectedGradientDescent(estimator=object_detector, max_iter=2)
    x_test_adv = attack.generate(x=x_test, y=y_test)
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test_adv, x_test)