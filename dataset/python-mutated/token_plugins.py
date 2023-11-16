import logging
import os
import sys
import time
import re
import json
logger = logging.getLogger(__name__)

class BasePlugin:

    def __init__(self, src):
        if False:
            print('Hello World!')
        self.source = src

    def lookup(self, token):
        if False:
            return 10
        return None

class ReadOnlyTokenFile(BasePlugin):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self._targets = None

    def _load_targets(self):
        if False:
            for i in range(10):
                print('nop')
        if os.path.isdir(self.source):
            cfg_files = [os.path.join(self.source, f) for f in os.listdir(self.source)]
        else:
            cfg_files = [self.source]
        self._targets = {}
        index = 1
        for f in cfg_files:
            for line in [l.strip() for l in open(f).readlines()]:
                if line and (not line.startswith('#')):
                    try:
                        (tok, target) = re.split(':\\s', line)
                        self._targets[tok] = target.strip().rsplit(':', 1)
                    except ValueError:
                        logger.error('Syntax error in %s on line %d' % (self.source, index))
                index += 1

    def lookup(self, token):
        if False:
            print('Hello World!')
        if self._targets is None:
            self._load_targets()
        if token in self._targets:
            return self._targets[token]
        else:
            return None

class TokenFile(ReadOnlyTokenFile):

    def lookup(self, token):
        if False:
            i = 10
            return i + 15
        self._load_targets()
        return super().lookup(token)

class BaseTokenAPI(BasePlugin):

    def process_result(self, resp):
        if False:
            while True:
                i = 10
        (host, port) = resp.text.split(':')
        port = port.encode('ascii', 'ignore')
        return [host, port]

    def lookup(self, token):
        if False:
            while True:
                i = 10
        import requests
        resp = requests.get(self.source % token)
        if resp.ok:
            return self.process_result(resp)
        else:
            return None

class JSONTokenApi(BaseTokenAPI):

    def process_result(self, resp):
        if False:
            print('Hello World!')
        resp_json = resp.json()
        return (resp_json['host'], resp_json['port'])

class JWTTokenApi(BasePlugin):

    def lookup(self, token):
        if False:
            i = 10
            return i + 15
        try:
            from jwcrypto import jwt, jwk
            import json
            key = jwk.JWK()
            try:
                with open(self.source, 'rb') as key_file:
                    key_data = key_file.read()
            except Exception as e:
                logger.error('Error loading key file: %s' % str(e))
                return None
            try:
                key.import_from_pem(key_data)
            except:
                try:
                    key.import_key(k=key_data.decode('utf-8'), kty='oct')
                except:
                    logger.error('Failed to correctly parse key data!')
                    return None
            try:
                token = jwt.JWT(key=key, jwt=token)
                parsed_header = json.loads(token.header)
                if 'enc' in parsed_header:
                    token = jwt.JWT(key=key, jwt=token.claims)
                parsed = json.loads(token.claims)
                if 'nbf' in parsed:
                    if time.time() < parsed['nbf']:
                        logger.warning('Token can not be used yet!')
                        return None
                if 'exp' in parsed:
                    if time.time() > parsed['exp']:
                        logger.warning('Token has expired!')
                        return None
                return (parsed['host'], parsed['port'])
            except Exception as e:
                logger.error('Failed to parse token: %s' % str(e))
                return None
        except ImportError:
            logger.error("package jwcrypto not found, are you sure you've installed it correctly?")
            return None

class TokenRedis(BasePlugin):
    """Token plugin based on the Redis in-memory data store.

    The token source is in the format:

        host[:port[:db[:password]]]

    where port, db and password are optional. If port or db are left empty
    they will take its default value, ie. 6379 and 0 respectively.

    If your redis server is using the default port (6379) then you can use:

        my-redis-host

    In case you need to authenticate with the redis server and you are using
    the default database and port you can use:

        my-redis-host:::verysecretpass

    In the more general case you will use:

        my-redis-host:6380:1:verysecretpass

    The TokenRedis plugin expects the format of the target in one of these two
    formats:

    - JSON

        {"host": "target-host:target-port"}

    - Plain text

        target-host:target-port

    Prepare data with:

        redis-cli set my-token '{"host": "127.0.0.1:5000"}'

    Verify with:

        redis-cli --raw get my-token

    Spawn a test "server" using netcat

        nc -l 5000 -v

    Note: This Token Plugin depends on the 'redis' module, so you have
    to install it before using this plugin:

          pip install redis
    """

    def __init__(self, src):
        if False:
            for i in range(10):
                print('nop')
        try:
            import redis
        except ImportError:
            logger.error('Unable to load redis module')
            sys.exit()
        self._port = 6379
        self._db = 0
        self._password = None
        try:
            fields = src.split(':')
            if len(fields) == 1:
                self._server = fields[0]
            elif len(fields) == 2:
                (self._server, self._port) = fields
                if not self._port:
                    self._port = 6379
            elif len(fields) == 3:
                (self._server, self._port, self._db) = fields
                if not self._port:
                    self._port = 6379
                if not self._db:
                    self._db = 0
            elif len(fields) == 4:
                (self._server, self._port, self._db, self._password) = fields
                if not self._port:
                    self._port = 6379
                if not self._db:
                    self._db = 0
                if not self._password:
                    self._password = None
            else:
                raise ValueError
            self._port = int(self._port)
            self._db = int(self._db)
            logger.info('TokenRedis backend initilized (%s:%s)' % (self._server, self._port))
        except ValueError:
            logger.error("The provided --token-source='%s' is not in the expected format <host>[:<port>[:<db>[:<password>]]]" % src)
            sys.exit()

    def lookup(self, token):
        if False:
            return 10
        try:
            import redis
        except ImportError:
            logger.error("package redis not found, are you sure you've installed them correctly?")
            sys.exit()
        logger.info("resolving token '%s'" % token)
        client = redis.Redis(host=self._server, port=self._port, db=self._db, password=self._password)
        stuff = client.get(token)
        if stuff is None:
            return None
        else:
            responseStr = stuff.decode('utf-8').strip()
            logger.debug('response from redis : %s' % responseStr)
            if responseStr.startswith('{'):
                try:
                    combo = json.loads(responseStr)
                    (host, port) = combo['host'].split(':')
                except ValueError:
                    logger.error('Unable to decode JSON token: %s' % responseStr)
                    return None
                except KeyError:
                    logger.error("Unable to find 'host' key in JSON token: %s" % responseStr)
                    return None
            elif re.match('\\S+:\\S+', responseStr):
                (host, port) = responseStr.split(':')
            else:
                logger.error('Unable to parse token: %s' % responseStr)
                return None
            logger.debug('host: %s, port: %s' % (host, port))
            return [host, port]

class UnixDomainSocketDirectory(BasePlugin):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self._dir_path = os.path.abspath(self.source)

    def lookup(self, token):
        if False:
            for i in range(10):
                print('nop')
        try:
            import stat
            if not os.path.isdir(self._dir_path):
                return None
            uds_path = os.path.abspath(os.path.join(self._dir_path, token))
            if not uds_path.startswith(self._dir_path):
                return None
            if not os.path.exists(uds_path):
                return None
            if not stat.S_ISSOCK(os.stat(uds_path).st_mode):
                return None
            return ['unix_socket', uds_path]
        except Exception as e:
            logger.error('Error finding unix domain socket: %s' % str(e))
            return None