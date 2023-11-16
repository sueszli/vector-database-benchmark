from __future__ import annotations
import logging
import os
import requests
from playsound import playsound
from autogpt.core.configuration import SystemConfiguration, UserConfigurable
from autogpt.speech.base import VoiceBase
logger = logging.getLogger(__name__)

class StreamElementsConfig(SystemConfiguration):
    voice: str = UserConfigurable(default='Brian')

class StreamElementsSpeech(VoiceBase):
    """Streamelements speech module for autogpt"""

    def _setup(self, config: StreamElementsConfig) -> None:
        if False:
            while True:
                i = 10
        'Setup the voices, API key, etc.'
        self.config = config

    def _speech(self, text: str, voice: str, _: int=0) -> bool:
        if False:
            while True:
                i = 10
        voice = self.config.voice
        'Speak text using the streamelements API\n\n        Args:\n            text (str): The text to speak\n            voice (str): The voice to use\n\n        Returns:\n            bool: True if the request was successful, False otherwise\n        '
        tts_url = f'https://api.streamelements.com/kappa/v2/speech?voice={voice}&text={text}'
        response = requests.get(tts_url)
        if response.status_code == 200:
            with open('speech.mp3', 'wb') as f:
                f.write(response.content)
            playsound('speech.mp3')
            os.remove('speech.mp3')
            return True
        else:
            logger.error('Request failed with status code: %s, response content: %s', response.status_code, response.content)
            return False