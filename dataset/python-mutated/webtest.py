import logging
from flask import abort, after_this_request
from errbot import BotPlugin
from errbot.core_plugins.webserver import webhook
log = logging.getLogger(__name__)

class WebTest(BotPlugin):

    @webhook
    def webhook1(self, payload):
        if False:
            while True:
                i = 10
        log.debug(str(payload))
        return str(payload)

    @webhook('/custom_webhook')
    def webhook2(self, payload):
        if False:
            return 10
        log.debug(str(payload))
        return str(payload)

    @webhook('/form', form_param='form')
    def webhook3(self, payload):
        if False:
            return 10
        log.debug(str(payload))
        return str(payload)

    @webhook('/custom_form', form_param='form')
    def webhook4(self, payload):
        if False:
            print('Hello World!')
        log.debug(str(payload))
        return str(payload)

    @webhook('/raw', raw=True)
    def webhook5(self, payload):
        if False:
            for i in range(10):
                print('nop')
        log.debug(str(payload))
        return str(type(payload))

    @webhook
    def webhook6(self, payload):
        if False:
            for i in range(10):
                print('nop')
        log.debug(str(payload))

        @after_this_request
        def add_header(response):
            if False:
                return 10
            response.headers['X-Powered-By'] = 'Errbot'
            return response
        return str(payload)

    @webhook
    def webhook7(self, payload):
        if False:
            print('Hello World!')
        abort(403, 'Forbidden')
    webhook8 = webhook('/lambda')(lambda x, y: str(x) + str(y))

    @webhook(raw=True)
    def raw2(self, payload):
        if False:
            print('Hello World!')
        log.debug(str(payload))
        return str(type(payload))