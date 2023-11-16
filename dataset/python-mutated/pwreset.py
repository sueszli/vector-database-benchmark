import logging
import os
import simplejson as json
from itsdangerous.url_safe import URLSafeTimedSerializer
from itsdangerous.exc import SignatureExpired, BadTimeSignature, BadSignature
from jadi import component
import aj
from aj.auth import AuthenticationService
from aj.api.mail import Mail
from aj.api.http import HttpMasterMiddleware
SECRET_FILE = '/etc/ajenti/.secret'

@component(HttpMasterMiddleware)
class PasswordResetMiddleware(HttpMasterMiddleware):

    def __init__(self, context):
        if False:
            return 10
        self.context = context
        self.auth_provider = AuthenticationService.get(self.context).get_provider()
        self.notifications = Mail()
        self.ensure_secret_key()

    def handle(self, http_context):
        if False:
            for i in range(10):
                print('nop')
        if http_context.path == '/api/master/send_password_reset':
            return self.send_password_reset(http_context)
        if http_context.path == '/api/master/check_password_serial':
            return self.check_serial(http_context)
        if http_context.path == '/api/master/update_password':
            return self.update_password(http_context)
        return None

    def ensure_secret_key(self):
        if False:
            print('Hello World!')
        '\n        Generate secret key and provide basic length test.\n        '
        if not os.path.isfile(SECRET_FILE):
            logging.info('No secret found, generating new secret key')
            with open(SECRET_FILE, 'w') as f:
                f.write(os.urandom(16).hex())
        else:
            with open(SECRET_FILE, 'r') as f:
                secret = f.read().strip('\n')
            if len(secret) < 32:
                logging.warning('Secret key is too weak, you need at least 32 chars')
        os.chmod(SECRET_FILE, 384)

    def send_password_reset(self, http_context):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sends upstream a request to check if a given email exists, in order to\n        send a password reset link per email.\n\n        :param http_context: HttpContext\n        :type http_context: HttpContext\n        '
        mail = json.loads(http_context.body.decode())['mail']
        username = self.auth_provider.check_mail(mail)
        if username:
            with open(SECRET_FILE, 'r') as f:
                secret = f.read().strip('\n')
                serializer = URLSafeTimedSerializer(secret)
            serial = serializer.dumps({'user': username, 'email': mail})
            if aj.config.data['trusted_domains']:
                base_url = aj.config.data['trusted_domains'][0]
            else:
                origin = http_context.env['SERVER_NAME']
                port = ''
                if http_context.env['SERVER_PORT'] not in [80, 443]:
                    port = f":{http_context.env['SERVER_PORT']}"
                base_url = f"{http_context.env['wsgi.url_scheme']}://{origin}{port}"
            link = f'{base_url}/view/reset_password/{serial}'
            self.notifications.send_password_reset(mail, link)

    def check_serial(self, http_context):
        if False:
            print('Hello World!')
        '\n        Check if the serial in a given reset link is valid\n\n        :param http_context: HttpContext\n        :type http_context: HttpContext\n        '
        serial = json.loads(http_context.body.decode())['serial']
        if serial:
            try:
                with open(SECRET_FILE, 'r') as f:
                    secret = f.read().strip('\n')
                    serializer = URLSafeTimedSerializer(secret)
                serializer.loads(serial, max_age=900)
                http_context.respond_ok()
                return [b'200 OK']
            except (SignatureExpired, BadTimeSignature, BadSignature) as err:
                logging.warning('Password reset link not valid or expired')
                http_context.respond_not_found()
                return [b'Link not valid']
        return False

    def update_password(self, http_context):
        if False:
            i = 10
            return i + 15
        "\n        Update user's password in the auth provider.\n\n        :param http_context: HttpContext\n        :type http_context: HttpContext\n        "
        serial = json.loads(http_context.body.decode())['serial']
        password = json.loads(http_context.body.decode())['password']
        if serial and password:
            with open(SECRET_FILE, 'r') as f:
                secret = f.read().strip('\n')
                serializer = URLSafeTimedSerializer(secret)
            user = serializer.loads(serial, max_age=900)['user']
            auth_provider = AuthenticationService.get(self.context).get_provider()
            answer = auth_provider.update_password(user, password)
            if not answer:
                http_context.respond_forbidden()
                return [b'403 Forbidden']
            else:
                http_context.respond_ok()
                return [b'200 OK']