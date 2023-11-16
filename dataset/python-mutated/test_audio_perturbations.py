from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import pytest
import os
from art.attacks.poisoning.perturbations.audio_perturbations import CacheToneTrigger, CacheAudioTrigger
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.mark.framework_agnostic
def test_insert_tone_trigger(art_warning):
    if False:
        while True:
            i = 10
    try:
        trigger = CacheToneTrigger(sampling_rate=16000)
        audio = trigger.insert(x=np.zeros(3200))
        assert audio.shape == (3200,)
        assert np.max(audio) != 0
        assert np.max(np.abs(audio)) <= 1.0
        trigger = CacheToneTrigger(sampling_rate=16000, frequency=16000, duration=0.2, scale=0.5)
        audio = trigger.insert(x=np.zeros(3200))
        assert audio.shape == (3200,)
        assert np.max(audio) != 0
        audio = trigger.insert(x=np.zeros((10, 3200)))
        assert audio.shape == (10, 3200)
        assert np.max(audio) != 0
        trigger = CacheToneTrigger(sampling_rate=16000, shift=10)
        audio = trigger.insert(x=np.zeros(3200))
        assert audio.shape == (3200,)
        assert np.max(audio) != 0
        assert np.sum(audio[:10]) == 0
        trigger = CacheToneTrigger(sampling_rate=16000, random=True)
        audio = trigger.insert(x=np.zeros((10, 3200)))
        assert audio.shape == (10, 3200)
        assert np.max(audio) != 0
        with pytest.raises(ValueError):
            trigger = CacheToneTrigger(sampling_rate=16000, duration=0.3)
            _ = trigger.insert(x=np.zeros(3200))
        with pytest.raises(ValueError):
            trigger = CacheToneTrigger(sampling_rate=16000, duration=0.2, shift=5)
            _ = trigger.insert(x=np.zeros(3200))
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_insert_audio_trigger(art_warning):
    if False:
        return 10
    file_path = os.path.join(os.getcwd(), 'utils/data/backdoors/cough_trigger.wav')
    try:
        trigger = CacheAudioTrigger(sampling_rate=16000, backdoor_path=file_path)
        audio = trigger.insert(x=np.zeros(32000))
        assert audio.shape == (32000,)
        assert np.max(audio) != 0
        assert np.max(np.abs(audio)) <= 1.0
        trigger = CacheAudioTrigger(sampling_rate=16000, backdoor_path=file_path, duration=0.8, scale=0.5)
        audio = trigger.insert(x=np.zeros(32000))
        assert audio.shape == (32000,)
        assert np.max(audio) != 0
        trigger = CacheAudioTrigger(sampling_rate=16000, backdoor_path=file_path)
        audio = trigger.insert(x=np.zeros((10, 16000)))
        assert audio.shape == (10, 16000)
        assert np.max(audio) != 0
        trigger = CacheAudioTrigger(sampling_rate=16000, backdoor_path=file_path, shift=10)
        audio = trigger.insert(x=np.zeros(32000))
        assert audio.shape == (32000,)
        assert np.max(audio) != 0
        assert np.sum(audio[:10]) == 0
        trigger = CacheAudioTrigger(sampling_rate=16000, backdoor_path=file_path, random=True)
        audio = trigger.insert(x=np.zeros((10, 32000)))
        assert audio.shape == (10, 32000)
        assert np.max(audio) != 0
        with pytest.raises(ValueError):
            trigger = CacheAudioTrigger(sampling_rate=16000, backdoor_path=file_path)
            _ = trigger.insert(x=np.zeros(15000))
        with pytest.raises(ValueError):
            trigger = CacheAudioTrigger(sampling_rate=16000, backdoor_path=file_path, duration=1, shift=5)
            _ = trigger.insert(x=np.zeros(16000))
    except ARTTestException as e:
        art_warning(e)