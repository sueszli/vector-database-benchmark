import os
import ssl
import sys
import cherrypy
import plexpy
if plexpy.PYTHON2:
    import logger
    import webauth
    from helpers import create_https_certificates
    from webserve import WebInterface, BaseRedirect
else:
    from plexpy import logger
    from plexpy import webauth
    from plexpy.helpers import create_https_certificates
    from plexpy.webserve import WebInterface, BaseRedirect

def start():
    if False:
        for i in range(10):
            print('nop')
    logger.info('Tautulli WebStart :: Initializing Tautulli web server...')
    web_config = {'http_port': plexpy.HTTP_PORT, 'http_host': plexpy.CONFIG.HTTP_HOST, 'http_root': plexpy.CONFIG.HTTP_ROOT, 'http_environment': plexpy.CONFIG.HTTP_ENVIRONMENT, 'http_proxy': plexpy.CONFIG.HTTP_PROXY, 'enable_https': plexpy.CONFIG.ENABLE_HTTPS, 'https_cert': plexpy.CONFIG.HTTPS_CERT, 'https_cert_chain': plexpy.CONFIG.HTTPS_CERT_CHAIN, 'https_key': plexpy.CONFIG.HTTPS_KEY, 'https_min_tls_version': plexpy.CONFIG.HTTPS_MIN_TLS_VERSION, 'http_username': plexpy.CONFIG.HTTP_USERNAME, 'http_password': plexpy.CONFIG.HTTP_PASSWORD, 'http_basic_auth': plexpy.CONFIG.HTTP_BASIC_AUTH}
    initialize(web_config)

def stop():
    if False:
        return 10
    logger.info('Tautulli WebStart :: Stopping Tautulli web server...')
    cherrypy.engine.exit()

def restart():
    if False:
        i = 10
        return i + 15
    logger.info('Tautulli WebStart :: Restarting Tautulli web server...')
    stop()
    start()

def initialize(options):
    if False:
        while True:
            i = 10
    enable_https = options['enable_https']
    https_cert = options['https_cert']
    https_cert_chain = options['https_cert_chain']
    https_key = options['https_key']
    if enable_https:
        if plexpy.CONFIG.HTTPS_CREATE_CERT and (not (https_cert and os.path.exists(https_cert)) or not (https_key and os.path.exists(https_key))):
            if not create_https_certificates(https_cert, https_key):
                logger.warn('Tautulli WebStart :: Unable to create certificate and key. Disabling HTTPS')
                enable_https = False
        if not (os.path.exists(https_cert) and os.path.exists(https_key)):
            logger.warn('Tautulli WebStart :: Disabled HTTPS because of missing certificate and key.')
            enable_https = False
    options_dict = {'server.socket_port': options['http_port'], 'server.socket_host': options['http_host'], 'environment': options['http_environment'], 'server.thread_pool': plexpy.CONFIG.HTTP_THREAD_POOL, 'server.max_request_body_size': 1073741824, 'server.socket_timeout': 60, 'tools.encode.on': True, 'tools.encode.encoding': 'utf-8', 'tools.decode.on': True}
    if plexpy.DEV:
        options_dict['environment'] = 'test_suite'
        options_dict['engine.autoreload.on'] = True
    if enable_https:
        context = ssl.create_default_context(purpose=ssl.Purpose.CLIENT_AUTH, cafile=https_cert_chain)
        min_tls_version = options['https_min_tls_version'].replace('.', '_')
        context.minimum_version = getattr(ssl.TLSVersion, min_tls_version, ssl.TLSVersion.TLSv1_2)
        logger.debug('Tautulli WebStart :: Minimum TLS version set to %s.', context.minimum_version.name)
        context.load_cert_chain(https_cert, https_key)
        options_dict['server.ssl_context'] = context
        options_dict['server.ssl_certificate'] = https_cert
        options_dict['server.ssl_certificate_chain'] = https_cert_chain
        options_dict['server.ssl_private_key'] = https_key
        protocol = 'https'
    else:
        protocol = 'http'
    if options['http_proxy']:
        cherrypy.tools.proxy = cherrypy.Tool('before_handler', proxy, priority=1)
    if options['http_password']:
        login_allowed = ["Tautulli admin (username is '%s')" % options['http_username']]
        if plexpy.CONFIG.HTTP_PLEX_ADMIN:
            login_allowed.append('Plex admin')
        logger.info('Tautulli WebStart :: Web server authentication is enabled: %s.', ' and '.join(login_allowed))
        if options['http_basic_auth']:
            plexpy.AUTH_ENABLED = False
            basic_auth_enabled = True
        else:
            plexpy.AUTH_ENABLED = True
            basic_auth_enabled = False
            cherrypy.tools.auth = cherrypy.Tool('before_handler', webauth.check_auth, priority=2)
    else:
        logger.warn('Tautulli WebStart :: Web server authentication is disabled!')
        plexpy.AUTH_ENABLED = False
        basic_auth_enabled = False
    if options['http_root'].strip('/'):
        plexpy.HTTP_ROOT = options['http_root'] = '/' + str(options['http_root'].strip('/')) + '/'
    else:
        plexpy.HTTP_ROOT = options['http_root'] = '/'
    logger.info('Tautulli WebStart :: Thread Pool Size: %d.', plexpy.CONFIG.HTTP_THREAD_POOL)
    cherrypy.config.update(options_dict)
    conf = {'/': {'engine.timeout_monitor.on': False, 'tools.staticdir.root': os.path.join(plexpy.PROG_DIR, 'data'), 'tools.proxy.on': bool(options['http_proxy']), 'tools.gzip.on': True, 'tools.gzip.mime_types': ['text/html', 'text/plain', 'text/css', 'text/javascript', 'application/json', 'application/javascript'], 'tools.auth.on': plexpy.AUTH_ENABLED, 'tools.auth_basic.on': basic_auth_enabled, 'tools.auth_basic.realm': 'Tautulli web server', 'tools.auth_basic.checkpassword': cherrypy.lib.auth_basic.checkpassword_dict({options['http_username']: options['http_password']})}, '/api': {'tools.auth_basic.on': False}, '/status': {'tools.auth_basic.on': False}, '/interfaces': {'tools.staticdir.on': True, 'tools.staticdir.dir': 'interfaces', 'tools.caching.on': True, 'tools.caching.force': True, 'tools.caching.delay': 0, 'tools.expires.on': True, 'tools.expires.secs': 60 * 60 * 24 * 30, 'tools.sessions.on': False, 'tools.auth.on': False}, '/images': {'tools.staticdir.on': True, 'tools.staticdir.dir': 'interfaces/default/images', 'tools.staticdir.content_types': {'svg': 'image/svg+xml'}, 'tools.caching.on': True, 'tools.caching.force': True, 'tools.caching.delay': 0, 'tools.expires.on': True, 'tools.expires.secs': 60 * 60 * 24 * 30, 'tools.sessions.on': False, 'tools.auth.on': False}, '/css': {'tools.staticdir.on': True, 'tools.staticdir.dir': 'interfaces/default/css', 'tools.caching.on': True, 'tools.caching.force': True, 'tools.caching.delay': 0, 'tools.expires.on': True, 'tools.expires.secs': 60 * 60 * 24 * 30, 'tools.sessions.on': False, 'tools.auth.on': False}, '/fonts': {'tools.staticdir.on': True, 'tools.staticdir.dir': 'interfaces/default/fonts', 'tools.caching.on': True, 'tools.caching.force': True, 'tools.caching.delay': 0, 'tools.expires.on': True, 'tools.expires.secs': 60 * 60 * 24 * 30, 'tools.sessions.on': False, 'tools.auth.on': False}, '/js': {'tools.staticdir.on': True, 'tools.staticdir.dir': 'interfaces/default/js', 'tools.caching.on': True, 'tools.caching.force': True, 'tools.caching.delay': 0, 'tools.expires.on': True, 'tools.expires.secs': 60 * 60 * 24 * 30, 'tools.sessions.on': False, 'tools.auth.on': False}, '/cache': {'tools.staticdir.on': True, 'tools.staticdir.dir': plexpy.CONFIG.CACHE_DIR, 'tools.caching.on': True, 'tools.caching.force': True, 'tools.caching.delay': 0, 'tools.expires.on': True, 'tools.expires.secs': 60 * 60 * 24 * 30, 'tools.sessions.on': False, 'tools.auth.on': False}, '/pms_image_proxy': {'tools.caching.on': True, 'tools.caching.force': True, 'tools.caching.delay': 0, 'tools.expires.on': True, 'tools.expires.secs': 60 * 60 * 24 * 30, 'tools.auth.on': False, 'tools.sessions.on': False}, '/favicon.ico': {'tools.staticfile.on': True, 'tools.staticfile.filename': os.path.abspath(os.path.join(plexpy.PROG_DIR, 'data/interfaces/default/images/favicon/favicon.ico')), 'tools.caching.on': True, 'tools.caching.force': True, 'tools.caching.delay': 0, 'tools.expires.on': True, 'tools.expires.secs': 60 * 60 * 24 * 30, 'tools.sessions.on': False, 'tools.auth.on': False}}
    cherrypy.tree.mount(WebInterface(), options['http_root'], config=conf)
    if plexpy.HTTP_ROOT != '/':
        cherrypy.tree.mount(BaseRedirect(), '/')
    try:
        logger.info('Tautulli WebStart :: Starting Tautulli web server on %s://%s:%d%s', protocol, options['http_host'], options['http_port'], options['http_root'])
        if not plexpy.DEV:
            cherrypy.server.start()
        else:
            cherrypy.engine.signals.subscribe()
            cherrypy.engine.start()
            cherrypy.engine.block()
    except IOError as e:
        logger.error('Tautulli WebStart :: Failed to start Tautulli: %s', e)
        plexpy.alert_message('Failed to start Tautulli: %s' % e)
        sys.exit(1)
    cherrypy.server.wait()

def proxy():
    if False:
        for i in range(10):
            print('nop')
    local = 'X-Forwarded-Host'
    if not cherrypy.request.headers.get('X-Forwarded-Host'):
        if cherrypy.request.headers.get('X-Host'):
            local = 'X-Host'
        elif cherrypy.request.headers.get('Origin'):
            local = 'Origin'
        elif cherrypy.request.headers.get('Host'):
            local = 'Host'
    cherrypy.lib.cptools.proxy(local=local)