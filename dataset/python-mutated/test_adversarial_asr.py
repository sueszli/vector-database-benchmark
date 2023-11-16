from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import pytest
from numpy.testing import assert_array_equal
from art.attacks.attack import EvasionAttack
from art.attacks.evasion.adversarial_asr import CarliniWagnerASR
from art.attacks.evasion.imperceptible_asr.imperceptible_asr import ImperceptibleASR
from art.estimators.estimator import BaseEstimator, LossGradientsMixin, NeuralNetworkMixin
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

class TestImperceptibleASR:
    """
    Test the ImperceptibleASR attack.
    """

    @pytest.mark.framework_agnostic
    def test_is_subclass(self, art_warning):
        if False:
            for i in range(10):
                print('nop')
        try:
            assert issubclass(CarliniWagnerASR, (ImperceptibleASR, EvasionAttack))
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_framework('tensorflow1', 'mxnet', 'kerastf', 'non_dl_frameworks')
    def test_implements_abstract_methods(self, art_warning, asr_dummy_estimator):
        if False:
            return 10
        try:
            CarliniWagnerASR(estimator=asr_dummy_estimator())
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.framework_agnostic
    def test_classifier_type_check_fail(self, art_warning):
        if False:
            while True:
                i = 10
        try:
            backend_test_classifier_type_check_fail(CarliniWagnerASR, [NeuralNetworkMixin, LossGradientsMixin, BaseEstimator, SpeechRecognizerMixin])
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_framework('tensorflow1', 'mxnet', 'kerastf', 'non_dl_frameworks')
    def test_generate_batch(self, art_warning, mocker, asr_dummy_estimator, audio_data):
        if False:
            while True:
                i = 10
        try:
            (test_input, test_target) = audio_data
            mocker.patch.object(CarliniWagnerASR, '_create_adversarial', return_value=test_input)
            carlini_asr = CarliniWagnerASR(estimator=asr_dummy_estimator())
            adversarial = carlini_asr._generate_batch(test_input, test_target)
            carlini_asr._create_adversarial.assert_called()
            for (a, t) in zip(adversarial, test_input):
                assert_array_equal(a, t)
        except ARTTestException as e:
            art_warning(e)