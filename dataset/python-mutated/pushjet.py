from twisted.internet import defer
from twisted.python import log as twlog
from buildbot.process.results import CANCELLED
from buildbot.process.results import EXCEPTION
from buildbot.process.results import FAILURE
from buildbot.process.results import SUCCESS
from buildbot.process.results import WARNINGS
from buildbot.reporters.base import ReporterBase
from buildbot.reporters.generators.build import BuildStatusGenerator
from buildbot.reporters.message import MessageFormatter
from buildbot.util import httpclientservice
from .utils import merge_reports_prop
from .utils import merge_reports_prop_take_first
ENCODING = 'utf8'
LEVELS = {CANCELLED: 'cancelled', EXCEPTION: 'exception', FAILURE: 'failing', SUCCESS: 'passing', WARNINGS: 'warnings'}
DEFAULT_MSG_TEMPLATE = 'The Buildbot has detected a <a href="{{ build_url }}">{{ status_detected }}</a>' + 'of <i>{{ buildername }}</i> while building {{ projects }} on {{ workername }}.'

class PushjetNotifier(ReporterBase):

    def checkConfig(self, secret, levels=None, base_url='https://api.pushjet.io', generators=None):
        if False:
            while True:
                i = 10
        if generators is None:
            generators = self._create_default_generators()
        super().checkConfig(generators=generators)
        httpclientservice.HTTPClientService.checkAvailable(self.__class__.__name__)

    @defer.inlineCallbacks
    def reconfigService(self, secret, levels=None, base_url='https://api.pushjet.io', generators=None):
        if False:
            i = 10
            return i + 15
        secret = (yield self.renderSecrets(secret))
        if generators is None:
            generators = self._create_default_generators()
        yield super().reconfigService(generators=generators)
        self.secret = secret
        if levels is None:
            self.levels = {}
        else:
            self.levels = levels
        self._http = (yield httpclientservice.HTTPClientService.getService(self.master, base_url))

    def _create_default_generators(self):
        if False:
            while True:
                i = 10
        formatter = MessageFormatter(template_type='html', template=DEFAULT_MSG_TEMPLATE)
        return [BuildStatusGenerator(message_formatter=formatter)]

    def sendMessage(self, reports):
        if False:
            return 10
        body = merge_reports_prop(reports, 'body')
        subject = merge_reports_prop_take_first(reports, 'subject')
        results = merge_reports_prop(reports, 'results')
        worker = merge_reports_prop_take_first(reports, 'worker')
        msg = {'message': body, 'title': subject}
        level = self.levels.get(LEVELS[results] if worker is None else 'worker_missing')
        if level is not None:
            msg['level'] = level
        return self.sendNotification(msg)

    def sendNotification(self, params):
        if False:
            return 10
        twlog.msg('sending pushjet notification')
        params.update({'secret': self.secret})
        return self._http.post('/message', data=params)