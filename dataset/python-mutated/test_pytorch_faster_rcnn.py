from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import pytest
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.mark.only_with_platform('pytorch')
def test_predict(art_warning, get_pytorch_faster_rcnn):
    if False:
        while True:
            i = 10
    try:
        (object_detector, x_test, _) = get_pytorch_faster_rcnn
        result = object_detector.predict(x_test)
        assert list(result[0].keys()) == ['boxes', 'labels', 'scores']
        assert result[0]['boxes'].shape == (7, 4)
        expected_detection_boxes = np.asarray([4.4017954, 6.3090835, 22.128296, 27.570665])
        np.testing.assert_array_almost_equal(result[0]['boxes'][2, :], expected_detection_boxes, decimal=3)
        assert result[0]['scores'].shape == (7,)
        expected_detection_scores = np.asarray([0.3314798, 0.14125851, 0.13928168, 0.0996184, 0.08550017, 0.06690315, 0.05359321])
        np.testing.assert_array_almost_equal(result[0]['scores'][:10], expected_detection_scores, decimal=6)
        assert result[0]['labels'].shape == (7,)
        expected_detection_classes = np.asarray([72, 79, 1, 72, 78, 72, 82])
        np.testing.assert_array_almost_equal(result[0]['labels'][:10], expected_detection_classes, decimal=6)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('pytorch')
def test_fit(art_warning, get_pytorch_faster_rcnn):
    if False:
        print('Hello World!')
    try:
        (object_detector, x_test, y_test) = get_pytorch_faster_rcnn
        loss1 = object_detector.compute_loss(x=x_test, y=y_test)
        object_detector.fit(x_test, y_test, nb_epochs=1)
        loss2 = object_detector.compute_loss(x=x_test, y=y_test)
        assert loss1 != loss2
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('pytorch')
def test_loss_gradient(art_warning, get_pytorch_faster_rcnn):
    if False:
        i = 10
        return i + 15
    try:
        (object_detector, x_test, y_test) = get_pytorch_faster_rcnn
        grads = object_detector.loss_gradient(x_test, y_test)
        assert grads.shape == (2, 28, 28, 3)
        expected_gradients1 = np.asarray([[0.00046265591, 0.0012323459, 0.001391504], [-0.0003265806, -0.0036941725, -0.00045638453], [0.00078702159, -0.0033072452, 0.00030583731], [0.0010381485, -0.0020846087, 0.00023015277], [0.0021460971, -0.0013157589, 0.00035176644], [0.0033839934, 0.0013083456, 0.001615594], [0.0038621046, 0.0016645766, 0.0018313043], [0.0030887076, 0.0014632678, 0.0011174511], [0.0033404885, 0.0020578136, 0.00096874911], [0.0032202434, 0.00072660763, 0.00089162006], [0.0035761783, 0.0023615893, 0.00088510796], [0.0034721815, 0.0019500104, 0.00092907902], [0.0034767685, 0.0021154548, 0.00055654044], [0.003949258, 0.0035505455, 0.00065863604], [0.0039963769, 0.0040338552, 0.00039539216], [0.0022312226, 5.1399925e-06, -0.0010743635], [0.0023955442, 0.00067116896, -0.0012389944], [0.0019969011, -0.00045717746, -0.0015225793], [0.0018131963, -0.00077948131, -0.0016078206], [0.0014277012, -0.00077973347, -0.0013463887], [0.00073705515, -0.0011704378, -0.00098979671], [0.0001089974, -0.0012144407, -0.0011339665], [0.0001225489, -0.00047438752, -0.00088673591], [0.00070695346, 0.00072568876, -0.00025591519], [0.00050835893, 0.00026866698, 0.000227314], [-0.0005993275, -0.0011667561, -0.0004804465], [0.00040421321, 0.00031692928, -8.3296909e-05], [4.0506107e-05, -0.00031728629, -0.00044132984]])
        np.testing.assert_array_almost_equal(grads[0, 0, :, :], expected_gradients1, decimal=2)
        expected_gradients2 = np.asarray([[0.00047986404, 0.00077701372, 0.0011786318], [0.00073503907, -0.0023474507, -0.00039008856], [0.00041874062, -0.0025707064, -0.0011054531], [-0.0017942721, -0.003396845, -0.0014989552], [-0.0029697213, -0.0046922294, -0.0013162185], [-0.0031759157, -0.0098660104, -0.0047163852], [0.0018666144, -0.0028793041, -0.0031324378], [0.01055588, 0.0076373261, 0.0053013843], [0.00089815725, -0.010321697, 0.0014192325], [0.0085643278, 0.0030152409, 0.0020114987], [-0.0027870361, -0.011686913, -0.0070649502], [-0.0077482774, -0.0013334424, -0.0091927368], [-0.008148782, -0.003813382, -0.0043300558], [-0.00770067, -0.012594147, -0.0039680018], [-0.0095743872, -0.021007264, -0.0091963671], [-0.008677722, -0.017278835, -0.013328674], [-0.017368209, -0.023461722, -0.011538444], [-0.0046307812, -0.0057058665, 0.0013555109], [0.0048570461, -0.0058050654, 0.0081082489], [0.0064304657, 0.0028407066, 0.0087463465], [0.0050593228, 0.0014102085, 0.0052116364], [0.0025003455, -0.00060178695, 0.0020183939], [0.0021247163, 0.00047659015, 0.00075940741], [0.0013499497, 0.00062203623, 0.00012288829], [0.00028991612, -0.0004021629, -7.2287643e-05], [6.6898909e-05, -0.00063778006, -0.0003629486], [0.00053613615, 9.9137833e-05, -1.6657988e-05], [-3.9828232e-05, -0.0003845313, -0.00023702848]])
        np.testing.assert_array_almost_equal(grads[1, 0, :, :], expected_gradients2, decimal=2)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('pytorch')
def test_errors(art_warning):
    if False:
        return 10
    try:
        from art.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN
        with pytest.raises(ValueError):
            PyTorchFasterRCNN(clip_values=(1, 2), attack_losses=('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'))
        with pytest.raises(ValueError):
            PyTorchFasterRCNN(clip_values=(-1, 1), attack_losses=('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'))
        with pytest.raises(ValueError):
            PyTorchFasterRCNN(clip_values=(0, 1), attack_losses=('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'), preprocessing=(0, 1))
        from art.defences.postprocessor.rounded import Rounded
        post_def = Rounded()
        with pytest.raises(ValueError):
            PyTorchFasterRCNN(clip_values=(0, 1), attack_losses=('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'), postprocessing_defences=post_def)
    except ARTTestException as e:
        art_warning(e)

def test_preprocessing_defences(art_warning, get_pytorch_faster_rcnn):
    if False:
        return 10
    try:
        from art.defences.preprocessor.spatial_smoothing import SpatialSmoothing
        pre_def = SpatialSmoothing()
        (object_detector, x_test, y_test) = get_pytorch_faster_rcnn
        object_detector.set_params(preprocessing_defences=pre_def)
        grads = object_detector.loss_gradient(x=x_test, y=y_test)
        assert grads.shape == (2, 28, 28, 3)
    except ARTTestException as e:
        art_warning(e)

def test_compute_losses(art_warning, get_pytorch_faster_rcnn):
    if False:
        while True:
            i = 10
    try:
        (object_detector, x_test, y_test) = get_pytorch_faster_rcnn
        losses = object_detector.compute_losses(x=x_test, y=y_test)
        assert len(losses) == 4
    except ARTTestException as e:
        art_warning(e)

def test_compute_loss(art_warning, get_pytorch_faster_rcnn):
    if False:
        while True:
            i = 10
    try:
        (object_detector, x_test, y_test) = get_pytorch_faster_rcnn
        loss = object_detector.compute_loss(x=x_test, y=y_test)
        assert pytest.approx(0.84883332, abs=0.01) == float(loss)
    except ARTTestException as e:
        art_warning(e)

def test_pgd(art_warning, get_pytorch_faster_rcnn):
    if False:
        while True:
            i = 10
    try:
        from art.attacks.evasion import ProjectedGradientDescent
        (object_detector, x_test, y_test) = get_pytorch_faster_rcnn
        attack = ProjectedGradientDescent(estimator=object_detector, max_iter=2)
        x_test_adv = attack.generate(x=x_test, y=y_test)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test_adv, x_test)
    except ARTTestException as e:
        art_warning(e)