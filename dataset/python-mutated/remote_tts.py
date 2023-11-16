import abc
import re
from requests_futures.sessions import FuturesSession
from .tts import TTS
from mycroft.util import play_wav
from mycroft.util.log import LOG

class RemoteTTSException(Exception):
    pass

class RemoteTTSTimeoutException(RemoteTTSException):
    pass

class RemoteTTS(TTS):
    """
    Abstract class for a Remote TTS engine implementation.

    It provides a common logic to perform multiple requests by splitting the
    whole sentence into small ones.
    """

    def __init__(self, lang, config, url, api_path, validator):
        if False:
            while True:
                i = 10
        super(RemoteTTS, self).__init__(lang, config, validator)
        self.api_path = api_path
        self.auth = None
        self.url = config.get('url', url).rstrip('/')
        self.session = FuturesSession()

    def execute(self, sentence, ident=None, listen=False):
        if False:
            for i in range(10):
                print('nop')
        phrases = self.__get_phrases(sentence)
        if len(phrases) > 0:
            for req in self.__requests(phrases):
                try:
                    self.begin_audio()
                    self.__play(req)
                except Exception as e:
                    LOG.error(e.message)
                finally:
                    self.end_audio(listen)

    @staticmethod
    def __get_phrases(sentence):
        if False:
            while True:
                i = 10
        phrases = re.split('\\.+[\\s+|\\n]', sentence)
        phrases = [p.replace('\n', '').strip() for p in phrases]
        phrases = [p for p in phrases if len(p) > 0]
        return phrases

    def __requests(self, phrases):
        if False:
            for i in range(10):
                print('nop')
        reqs = []
        for p in phrases:
            reqs.append(self.__request(p))
        return reqs

    def __request(self, p):
        if False:
            return 10
        return self.session.get(self.url + self.api_path, params=self.build_request_params(p), timeout=10, verify=False, auth=self.auth)

    @abc.abstractmethod
    def build_request_params(self, sentence):
        if False:
            return 10
        pass

    def __play(self, req):
        if False:
            while True:
                i = 10
        resp = req.result()
        if resp.status_code == 200:
            self.__save(resp.content)
            play_wav(self.filename).communicate()
        else:
            LOG.error('%s Http Error: %s for url: %s' % (resp.status_code, resp.reason, resp.url))

    def __save(self, data):
        if False:
            while True:
                i = 10
        with open(self.filename, 'wb') as f:
            f.write(data)