from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import pytest
from art.defences.transformer.poisoning import STRIP
from art.estimators.classification import TensorFlowClassifier, TensorFlowV2Classifier
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.mark.framework_agnostic
def test_strip(art_warning, get_default_mnist_subset, image_dl_estimator):
    if False:
        i = 10
        return i + 15
    try:
        ((x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)) = get_default_mnist_subset
        (classifier, _) = image_dl_estimator()
        classifier.fit(x_train_mnist, y_train_mnist, nb_epochs=1)
        strip = STRIP(classifier)
        defense_cleanse = strip()
        defense_cleanse.mitigate(x_test_mnist)
        defense_cleanse.predict(x_test_mnist)
        stripped_classifier = strip.get_classifier()
        stripped_classifier._check_params()
        assert isinstance(stripped_classifier, TensorFlowV2Classifier) or isinstance(stripped_classifier, TensorFlowClassifier)
    except ARTTestException as e:
        art_warning(e)