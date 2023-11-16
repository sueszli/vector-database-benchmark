from __future__ import unicode_literals
from future.builtins import next
from future.builtins import object
from future.builtins import str
from backports import csv
from io import open, BytesIO
import base64
import json
import ssl as _ssl
import linecache
import os
import shutil
import sys
import threading
import zipfile
from future.moves.urllib.parse import urlencode
import cherrypy
from cherrypy.lib.static import serve_file, serve_fileobj, serve_download
from cherrypy._cperror import NotFound
from hashing_passwords import make_hash
from mako.lookup import TemplateLookup
import mako.template
import mako.exceptions
import certifi
import websocket
if sys.version_info >= (3, 6):
    import secrets
import plexpy
if plexpy.PYTHON2:
    import activity_pinger
    import activity_processor
    import common
    import config
    import database
    import datafactory
    import exporter
    import graphs
    import helpers
    import http_handler
    import libraries
    import log_reader
    import logger
    import newsletter_handler
    import newsletters
    import mobile_app
    import notification_handler
    import notifiers
    import plextv
    import plexivity_import
    import plexwatch_import
    import pmsconnect
    import users
    import versioncheck
    import web_socket
    import webstart
    from api2 import API2
    from helpers import checked, addtoapi, get_ip, create_https_certificates, build_datatables_json, sanitize_out
    from session import get_session_info, get_session_user_id, allow_session_user, allow_session_library
    from webauth import AuthController, requireAuth, member_of, check_auth, get_jwt_token
    if common.PLATFORM == 'Windows':
        import windows
    elif common.PLATFORM == 'Darwin':
        import macos
else:
    from plexpy import activity_pinger
    from plexpy import activity_processor
    from plexpy import common
    from plexpy import config
    from plexpy import database
    from plexpy import datafactory
    from plexpy import exporter
    from plexpy import graphs
    from plexpy import helpers
    from plexpy import http_handler
    from plexpy import libraries
    from plexpy import log_reader
    from plexpy import logger
    from plexpy import newsletter_handler
    from plexpy import newsletters
    from plexpy import mobile_app
    from plexpy import notification_handler
    from plexpy import notifiers
    from plexpy import plextv
    from plexpy import plexivity_import
    from plexpy import plexwatch_import
    from plexpy import pmsconnect
    from plexpy import users
    from plexpy import versioncheck
    from plexpy import web_socket
    from plexpy import webstart
    from plexpy.api2 import API2
    from plexpy.helpers import checked, addtoapi, get_ip, create_https_certificates, build_datatables_json, sanitize_out
    from plexpy.session import get_session_info, get_session_user_id, allow_session_user, allow_session_library
    from plexpy.webauth import AuthController, requireAuth, member_of, check_auth, get_jwt_token
    if common.PLATFORM == 'Windows':
        from plexpy import windows
    elif common.PLATFORM == 'Darwin':
        from plexpy import macos
TEMPLATE_LOOKUP = None

def serve_template(template_name, **kwargs):
    if False:
        print('Hello World!')
    global TEMPLATE_LOOKUP
    if TEMPLATE_LOOKUP is None:
        interface_dir = os.path.join(str(plexpy.PROG_DIR), 'data/interfaces/')
        template_dir = os.path.join(str(interface_dir), plexpy.CONFIG.INTERFACE)
        TEMPLATE_LOOKUP = TemplateLookup(directories=[template_dir], default_filters=['unicode', 'h'], error_handler=mako_error_handler)
    http_root = plexpy.HTTP_ROOT
    server_name = helpers.pms_name()
    cache_param = '?' + (plexpy.CURRENT_VERSION or common.RELEASE)
    _session = get_session_info()
    try:
        template = TEMPLATE_LOOKUP.get_template(template_name)
        return template.render(http_root=http_root, server_name=server_name, cache_param=cache_param, _session=_session, **kwargs)
    except Exception as e:
        logger.exception('WebUI :: Mako template render error: %s' % e)
        return mako.exceptions.html_error_template().render()

def mako_error_handler(context, error):
    if False:
        return 10
    'Decorate tracebacks when Mako errors happen.\n    Evil hack: walk the traceback frames, find compiled Mako templates,\n    stuff their (transformed) source into linecache.cache.\n    '
    rich_tb = mako.exceptions.RichTraceback(error)
    rich_iter = iter(rich_tb.traceback)
    tb = sys.exc_info()[-1]
    source = {}
    annotated = set()
    while tb is not None:
        cur_rich = next(rich_iter)
        f = tb.tb_frame
        co = f.f_code
        filename = co.co_filename
        lineno = tb.tb_lineno
        if filename.startswith('memory:'):
            lines = source.get(filename)
            if lines is None:
                info = mako.template._get_module_info(filename)
                lines = source[filename] = info.module_source.splitlines(True)
                linecache.cache[filename] = (None, None, lines, filename)
            if (filename, lineno) not in annotated:
                annotated.add((filename, lineno))
                extra = '    # {} line {} in {}:\n    # {}'.format(*cur_rich)
                lines[lineno - 1] += extra
        tb = tb.tb_next
    raise

class BaseRedirect(object):

    @cherrypy.expose
    def index(self):
        if False:
            return 10
        raise cherrypy.HTTPRedirect(plexpy.HTTP_ROOT)

    @cherrypy.expose
    def status(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        path = '/' + '/'.join(args) if args else ''
        query = '?' + urlencode(kwargs) if kwargs else ''
        raise cherrypy.HTTPRedirect(plexpy.HTTP_ROOT + 'status' + path + query)

class WebInterface(object):
    auth = AuthController()

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.interface_dir = os.path.join(str(plexpy.PROG_DIR), 'data/')

    @cherrypy.expose
    @requireAuth()
    def index(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if plexpy.CONFIG.FIRST_RUN_COMPLETE:
            raise cherrypy.HTTPRedirect(plexpy.HTTP_ROOT + 'home')
        else:
            raise cherrypy.HTTPRedirect(plexpy.HTTP_ROOT + 'welcome')

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def welcome(self, **kwargs):
        if False:
            i = 10
            return i + 15
        config = {'pms_identifier': plexpy.CONFIG.PMS_IDENTIFIER, 'pms_ip': plexpy.CONFIG.PMS_IP, 'pms_port': plexpy.CONFIG.PMS_PORT, 'pms_is_remote': plexpy.CONFIG.PMS_IS_REMOTE, 'pms_ssl': plexpy.CONFIG.PMS_SSL, 'pms_is_cloud': plexpy.CONFIG.PMS_IS_CLOUD, 'pms_name': helpers.pms_name(), 'logging_ignore_interval': plexpy.CONFIG.LOGGING_IGNORE_INTERVAL}
        if plexpy.CONFIG.FIRST_RUN_COMPLETE:
            plexpy.initialize_scheduler()
            raise cherrypy.HTTPRedirect(plexpy.HTTP_ROOT + 'home')
        else:
            return serve_template(template_name='welcome.html', title='Welcome', config=config)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def save_pms_token(self, token=None, client_id=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if token is not None:
            plexpy.CONFIG.PMS_TOKEN = token
        if client_id is not None:
            plexpy.CONFIG.PMS_CLIENT_ID = client_id
        plexpy.CONFIG.write()

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi('get_server_list')
    def discover(self, include_cloud=True, all_servers=True, **kwargs):
        if False:
            while True:
                i = 10
        ' Get all your servers that are published to Plex.tv.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    [{"clientIdentifier": "ds48g4r354a8v9byrrtr697g3g79w",\n                      "httpsRequired": "0",\n                      "ip": "xxx.xxx.xxx.xxx",\n                      "label": "Winterfell-Server",\n                      "local": "1",\n                      "port": "32400",\n                      "value": "xxx.xxx.xxx.xxx"\n                      },\n                     {...},\n                     {...}\n                     ]\n            ```\n        '
        include_cloud = not include_cloud == 'false'
        all_servers = not all_servers == 'false'
        plex_tv = plextv.PlexTV()
        servers_list = plex_tv.discover(include_cloud=include_cloud, all_servers=all_servers)
        if servers_list:
            return servers_list

    @cherrypy.expose
    @requireAuth()
    def home(self, **kwargs):
        if False:
            return 10
        config = {'home_sections': plexpy.CONFIG.HOME_SECTIONS, 'home_refresh_interval': plexpy.CONFIG.HOME_REFRESH_INTERVAL, 'pms_name': helpers.pms_name(), 'pms_is_cloud': plexpy.CONFIG.PMS_IS_CLOUD, 'update_show_changelog': plexpy.CONFIG.UPDATE_SHOW_CHANGELOG, 'first_run_complete': plexpy.CONFIG.FIRST_RUN_COMPLETE}
        return serve_template(template_name='index.html', title='Home', config=config)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @addtoapi()
    def get_date_formats(self, **kwargs):
        if False:
            while True:
                i = 10
        ' Get the date and time formats used by Tautulli.\n\n             ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    {"date_format": "YYYY-MM-DD",\n                     "time_format": "HH:mm",\n                     }\n            ```\n        '
        if plexpy.CONFIG.DATE_FORMAT:
            date_format = plexpy.CONFIG.DATE_FORMAT
        else:
            date_format = 'YYYY-MM-DD'
        if plexpy.CONFIG.TIME_FORMAT:
            time_format = plexpy.CONFIG.TIME_FORMAT
        else:
            time_format = 'HH:mm'
        formats = {'date_format': date_format, 'time_format': time_format}
        return formats

    @cherrypy.expose
    @requireAuth()
    def get_current_activity(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pms_connect = pmsconnect.PmsConnect(token=plexpy.CONFIG.PMS_TOKEN)
        result = pms_connect.get_current_activity()
        if result:
            return serve_template(template_name='current_activity.html', data=result)
        else:
            logger.warn('Unable to retrieve data for get_current_activity.')
            return serve_template(template_name='current_activity.html', data=None)

    @cherrypy.expose
    @requireAuth()
    def get_current_activity_instance(self, session_key=None, **kwargs):
        if False:
            print('Hello World!')
        pms_connect = pmsconnect.PmsConnect(token=plexpy.CONFIG.PMS_TOKEN)
        result = pms_connect.get_current_activity()
        if result:
            session = next((s for s in result['sessions'] if s['session_key'] == session_key), None)
            return serve_template(template_name='current_activity_instance.html', session=session)
        else:
            return serve_template(template_name='current_activity_instance.html', session=None)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def terminate_session(self, session_key='', session_id='', message='', **kwargs):
        if False:
            return 10
        ' Stop a streaming session.\n\n            ```\n            Required parameters:\n                session_key (int):          The session key of the session to terminate, OR\n                session_id (str):           The session id of the session to terminate\n\n            Optional parameters:\n                message (str):              A custom message to send to the client\n\n            Returns:\n                None\n            ```\n        '
        pms_connect = pmsconnect.PmsConnect()
        result = pms_connect.terminate_session(session_key=session_key, session_id=session_id, message=message)
        if isinstance(result, str):
            return {'result': 'error', 'message': 'Failed to terminate session: {}.'.format(result)}
        elif result is True:
            return {'result': 'success', 'message': 'Session terminated.'}
        else:
            return {'result': 'error', 'message': 'Failed to terminate session.'}

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def open_plex_xml(self, endpoint='', plextv=False, **kwargs):
        if False:
            return 10
        if helpers.bool_true(plextv):
            base_url = 'https://plex.tv'
        else:
            base_url = plexpy.CONFIG.PMS_URL_OVERRIDE or plexpy.CONFIG.PMS_URL
        if '{machine_id}' in endpoint:
            endpoint = endpoint.format(machine_id=plexpy.CONFIG.PMS_IDENTIFIER)
        url = base_url + endpoint + ('?' + urlencode(kwargs) if kwargs else '')
        return serve_template(template_name='xml_shortcut.html', title='Plex XML', url=url)

    @cherrypy.expose
    @requireAuth()
    def home_stats(self, time_range=30, stats_type='plays', stats_count=10, **kwargs):
        if False:
            while True:
                i = 10
        data_factory = datafactory.DataFactory()
        stats_data = data_factory.get_home_stats(time_range=time_range, stats_type=stats_type, stats_count=stats_count)
        return serve_template(template_name='home_stats.html', title='Stats', data=stats_data)

    @cherrypy.expose
    @requireAuth()
    def library_stats(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        data_factory = datafactory.DataFactory()
        library_cards = plexpy.CONFIG.HOME_LIBRARY_CARDS
        stats_data = data_factory.get_library_stats(library_cards=library_cards)
        return serve_template(template_name='library_stats.html', title='Library Stats', data=stats_data)

    @cherrypy.expose
    @requireAuth()
    def get_recently_added(self, count='0', media_type='', **kwargs):
        if False:
            while True:
                i = 10
        try:
            pms_connect = pmsconnect.PmsConnect()
            result = pms_connect.get_recently_added_details(count=count, media_type=media_type)
        except IOError as e:
            return serve_template(template_name='recently_added.html', data=None)
        if result and 'recently_added' in result:
            return serve_template(template_name='recently_added.html', data=result['recently_added'])
        else:
            logger.warn('Unable to retrieve data for get_recently_added.')
            return serve_template(template_name='recently_added.html', data=None)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def regroup_history(self, **kwargs):
        if False:
            return 10
        ' Regroup play history in the database.'
        threading.Thread(target=activity_processor.regroup_history).start()
        return {'result': 'success', 'message': 'Regrouping play history started. Check the logs to monitor any problems.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def delete_temp_sessions(self, **kwargs):
        if False:
            while True:
                i = 10
        ' Flush out all of the temporary sessions in the database.'
        result = database.delete_sessions()
        if result:
            return {'result': 'success', 'message': 'Temporary sessions flushed.'}
        else:
            return {'result': 'error', 'message': 'Flush sessions failed.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def delete_recently_added(self, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Flush out all of the recently added items in the database.'
        result = database.delete_recently_added()
        if result:
            return {'result': 'success', 'message': 'Recently added flushed.'}
        else:
            return {'result': 'error', 'message': 'Flush recently added failed.'}

    @cherrypy.expose
    @requireAuth()
    def libraries(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return serve_template(template_name='libraries.html', title='Libraries')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @sanitize_out()
    @addtoapi('get_libraries_table')
    def get_library_list(self, grouping=None, **kwargs):
        if False:
            while True:
                i = 10
        ' Get the data on the Tautulli libraries table.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                grouping (int):                 0 or 1\n                order_column (str):             "library_thumb", "section_name", "section_type", "count", "parent_count",\n                                                "child_count", "last_accessed", "last_played", "plays", "duration"\n                order_dir (str):                "desc" or "asc"\n                start (int):                    Row to start from, 0\n                length (int):                   Number of items to return, 25\n                search (str):                   A string to search for, "Movies"\n\n            Returns:\n                json:\n                    {"draw": 1,\n                     "recordsTotal": 10,\n                     "recordsFiltered": 10,\n                     "data":\n                        [{"child_count": 3745,\n                          "content_rating": "TV-MA",\n                          "count": 62,\n                          "do_notify": 1,\n                          "do_notify_created": 1,\n                          "duration": 1578037,\n                          "guid": "com.plexapp.agents.thetvdb://121361/6/1?lang=en",\n                          "histroy_row_id": 1128,\n                          "is_active": 1,\n                          "keep_history": 1,\n                          "labels": [],\n                          "last_accessed": 1462693216,\n                          "last_played": "Game of Thrones - The Red Woman",\n                          "library_art": "/:/resources/show-fanart.jpg",\n                          "library_thumb": "/:/resources/show.png",\n                          "live": 0,\n                          "media_index": 1,\n                          "media_type": "episode",\n                          "originally_available_at": "2016-04-24",\n                          "parent_count": 240,\n                          "parent_media_index": 6,\n                          "parent_title": "",\n                          "plays": 772,\n                          "rating_key": 153037,\n                          "row_id": 1,\n                          "section_id": 2,\n                          "section_name": "TV Shows",\n                          "section_type": "Show",\n                          "server_id": "ds48g4r354a8v9byrrtr697g3g79w",\n                          "thumb": "/library/metadata/153036/thumb/1462175062",\n                          "year": 2016\n                          },\n                         {...},\n                         {...}\n                         ]\n                     }\n            ```\n        '
        if not kwargs.get('json_data'):
            dt_columns = [('library_thumb', False, False), ('section_name', True, True), ('section_type', True, True), ('count', True, True), ('parent_count', True, True), ('child_count', True, True), ('last_accessed', True, False), ('last_played', True, True), ('plays', True, False), ('duration', True, False)]
            kwargs['json_data'] = build_datatables_json(kwargs, dt_columns, 'section_name')
        grouping = helpers.bool_true(grouping, return_none=True)
        library_data = libraries.Libraries()
        library_list = library_data.get_datatables_list(kwargs=kwargs, grouping=grouping)
        return library_list

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @sanitize_out()
    @addtoapi('get_library_names')
    def get_library_sections(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Get a list of library sections and ids on the PMS.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    [{"section_id": 1, "section_name": "Movies", "section_type": "movie"},\n                     {"section_id": 7, "section_name": "Music", "section_type": "artist"},\n                     {"section_id": 2, "section_name": "TV Shows", "section_type": "show"},\n                     {...}\n                     ]\n            ```\n        '
        library_data = libraries.Libraries()
        result = library_data.get_sections()
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_library_sections.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def refresh_libraries_list(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Manually refresh the libraries list. '
        logger.info('Manual libraries list refresh requested.')
        result = libraries.refresh_libraries()
        if result:
            return {'result': 'success', 'message': 'Libraries list refreshed.'}
        else:
            return {'result': 'error', 'message': 'Unable to refresh libraries list.'}

    @cherrypy.expose
    @requireAuth()
    def library(self, section_id=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if not allow_session_library(section_id):
            raise cherrypy.HTTPRedirect(plexpy.HTTP_ROOT)
        config = {'get_file_sizes': plexpy.CONFIG.GET_FILE_SIZES, 'get_file_sizes_hold': plexpy.CONFIG.GET_FILE_SIZES_HOLD}
        if section_id:
            try:
                library_data = libraries.Libraries()
                library_details = library_data.get_details(section_id=section_id)
            except:
                logger.warn('Unable to retrieve library details for section_id %s ' % section_id)
                return serve_template(template_name='library.html', title='Library', data=None, config=config)
        else:
            logger.debug('Library page requested but no section_id received.')
            return serve_template(template_name='library.html', title='Library', data=None, config=config)
        return serve_template(template_name='library.html', title='Library', data=library_details, config=config)

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def edit_library_dialog(self, section_id=None, **kwargs):
        if False:
            print('Hello World!')
        if section_id:
            library_data = libraries.Libraries()
            result = library_data.get_details(section_id=section_id)
            status_message = ''
        else:
            result = None
            status_message = 'An error occured.'
        return serve_template(template_name='edit_library.html', title='Edit Library', data=result, server_id=plexpy.CONFIG.PMS_IDENTIFIER, status_message=status_message)

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    @addtoapi()
    def edit_library(self, section_id=None, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Update a library section on Tautulli.\n\n            ```\n            Required parameters:\n                section_id (str):           The id of the Plex library section\n                custom_thumb (str):         The URL for the custom library thumbnail\n                custom_art (str):           The URL for the custom library background art\n                keep_history (int):         0 or 1\n\n            Optional parameters:\n                None\n\n            Returns:\n                None\n            ```\n        '
        custom_thumb = kwargs.get('custom_thumb', '')
        custom_art = kwargs.get('custom_art', '')
        do_notify = kwargs.get('do_notify', 0)
        do_notify_created = kwargs.get('do_notify_created', 0)
        keep_history = kwargs.get('keep_history', 0)
        if section_id:
            try:
                library_data = libraries.Libraries()
                library_data.set_config(section_id=section_id, custom_thumb=custom_thumb, custom_art=custom_art, do_notify=do_notify, do_notify_created=do_notify_created, keep_history=keep_history)
                return 'Successfully updated library.'
            except:
                return 'Failed to update library.'

    @cherrypy.expose
    @requireAuth()
    def library_watch_time_stats(self, section_id=None, **kwargs):
        if False:
            print('Hello World!')
        if not allow_session_library(section_id):
            return serve_template(template_name='user_watch_time_stats.html', data=None, title='Watch Stats')
        if section_id:
            library_data = libraries.Libraries()
            result = library_data.get_watch_time_stats(section_id=section_id)
        else:
            result = None
        if result:
            return serve_template(template_name='user_watch_time_stats.html', data=result, title='Watch Stats')
        else:
            logger.warn('Unable to retrieve data for library_watch_time_stats.')
            return serve_template(template_name='user_watch_time_stats.html', data=None, title='Watch Stats')

    @cherrypy.expose
    @requireAuth()
    def library_user_stats(self, section_id=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if not allow_session_library(section_id):
            return serve_template(template_name='library_user_stats.html', data=None, title='Player Stats')
        if section_id:
            library_data = libraries.Libraries()
            result = library_data.get_user_stats(section_id=section_id)
        else:
            result = None
        if result:
            return serve_template(template_name='library_user_stats.html', data=result, title='Player Stats')
        else:
            logger.warn('Unable to retrieve data for library_user_stats.')
            return serve_template(template_name='library_user_stats.html', data=None, title='Player Stats')

    @cherrypy.expose
    @requireAuth()
    def library_recently_watched(self, section_id=None, limit='10', **kwargs):
        if False:
            print('Hello World!')
        if not allow_session_library(section_id):
            return serve_template(template_name='user_recently_watched.html', data=None, title='Recently Watched')
        if section_id:
            library_data = libraries.Libraries()
            result = library_data.get_recently_watched(section_id=section_id, limit=limit)
        else:
            result = None
        if result:
            return serve_template(template_name='user_recently_watched.html', data=result, title='Recently Watched')
        else:
            logger.warn('Unable to retrieve data for library_recently_watched.')
            return serve_template(template_name='user_recently_watched.html', data=None, title='Recently Watched')

    @cherrypy.expose
    @requireAuth()
    def library_recently_added(self, section_id=None, limit='10', **kwargs):
        if False:
            while True:
                i = 10
        if not allow_session_library(section_id):
            return serve_template(template_name='library_recently_added.html', data=None, title='Recently Added')
        if section_id:
            pms_connect = pmsconnect.PmsConnect()
            result = pms_connect.get_recently_added_details(section_id=section_id, count=limit)
        else:
            result = None
        if result and result['recently_added']:
            return serve_template(template_name='library_recently_added.html', data=result['recently_added'], title='Recently Added')
        else:
            logger.warn('Unable to retrieve data for library_recently_added.')
            return serve_template(template_name='library_recently_added.html', data=None, title='Recently Added')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_library_media_info(self, section_id=None, section_type=None, rating_key=None, refresh='', **kwargs):
        if False:
            return 10
        ' Get the data on the Tautulli media info tables.\n\n            ```\n            Required parameters:\n                section_id (str):               The id of the Plex library section, OR\n                rating_key (str):               The grandparent or parent rating key\n\n            Optional parameters:\n                section_type (str):             "movie", "show", "artist", "photo"\n                order_column (str):             "added_at", "sort_title", "container", "bitrate", "video_codec",\n                                                "video_resolution", "video_framerate", "audio_codec", "audio_channels",\n                                                "file_size", "last_played", "play_count"\n                order_dir (str):                "desc" or "asc"\n                start (int):                    Row to start from, 0\n                length (int):                   Number of items to return, 25\n                search (str):                   A string to search for, "Thrones"\n                refresh (str):                  "true" to refresh the media info table\n\n            Returns:\n                json:\n                    {"draw": 1,\n                     "last_refreshed": 1678734670,\n                     "recordsTotal": 82,\n                     "recordsFiltered": 82,\n                     "filtered_file_size": 2616760056742,\n                     "total_file_size": 2616760056742,\n                     "data":\n                        [{"added_at": "1403553078",\n                          "audio_channels": "",\n                          "audio_codec": "",\n                          "bitrate": "",\n                          "container": "",\n                          "file_size": 253660175293,\n                          "grandparent_rating_key": "",\n                          "last_played": 1462380698,\n                          "media_index": "1",\n                          "media_type": "show",\n                          "parent_media_index": "",\n                          "parent_rating_key": "",\n                          "play_count": 15,\n                          "rating_key": "1219",\n                          "section_id": 2,\n                          "section_type": "show",\n                          "sort_title": "Game of Thrones",\n                          "thumb": "/library/metadata/1219/thumb/1436265995",\n                          "title": "Game of Thrones",\n                          "video_codec": "",\n                          "video_framerate": "",\n                          "video_resolution": "",\n                          "year": "2011"\n                          },\n                         {...},\n                         {...}\n                         ]\n                     }\n            ```\n        '
        if not kwargs.get('json_data'):
            if kwargs.get('order_column') == 'title':
                kwargs['order_column'] = 'sort_title'
            dt_columns = [('added_at', True, False), ('sort_title', True, True), ('container', True, True), ('bitrate', True, True), ('video_codec', True, True), ('video_resolution', True, True), ('video_framerate', True, True), ('audio_codec', True, True), ('audio_channels', True, True), ('file_size', True, False), ('last_played', True, False), ('play_count', True, False)]
            kwargs['json_data'] = build_datatables_json(kwargs, dt_columns, 'sort_title')
        if helpers.bool_true(refresh):
            refresh = True
        else:
            refresh = False
        library_data = libraries.Libraries()
        result = library_data.get_datatables_media_info(section_id=section_id, section_type=section_type, rating_key=rating_key, refresh=refresh, kwargs=kwargs)
        return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @addtoapi('get_collections_table')
    def get_collections_list(self, section_id=None, **kwargs):
        if False:
            while True:
                i = 10
        ' Get the data on the Tautulli collections tables.\n\n            ```\n            Required parameters:\n                section_id (str):               The id of the Plex library section\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    {"draw": 1,\n                     "recordsTotal": 5,\n                     "data":\n                        [...]\n                     }\n            ```\n        '
        if not kwargs.get('json_data'):
            dt_columns = [('titleSort', True, True), ('collectionMode', True, True), ('collectionSort', True, True), ('childCount', True, False)]
            kwargs['json_data'] = build_datatables_json(kwargs, dt_columns, 'titleSort')
        result = libraries.get_collections_list(section_id=section_id, **kwargs)
        return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @addtoapi('get_playlists_table')
    def get_playlists_list(self, section_id=None, user_id=None, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Get the data on the Tautulli playlists tables.\n\n            ```\n            Required parameters:\n                section_id (str):               The section id of the Plex library, OR\n                user_id (str):                  The user id of the Plex user\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    {"draw": 1,\n                     "recordsTotal": 5,\n                     "data":\n                        [...]\n                     }\n            ```\n        '
        if not kwargs.get('json_data'):
            dt_columns = [('title', True, True), ('leafCount', True, True), ('duration', True, True)]
            kwargs['json_data'] = build_datatables_json(kwargs, dt_columns, 'title')
        result = libraries.get_playlists_list(section_id=section_id, user_id=user_id, **kwargs)
        return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def get_media_info_file_sizes(self, section_id=None, rating_key=None, **kwargs):
        if False:
            return 10
        get_file_sizes_hold = plexpy.CONFIG.GET_FILE_SIZES_HOLD
        section_ids = set(get_file_sizes_hold['section_ids'])
        rating_keys = set(get_file_sizes_hold['rating_keys'])
        section_id = helpers.cast_to_int(section_id)
        rating_key = helpers.cast_to_int(rating_key)
        if section_id and section_id not in section_ids or (rating_key and rating_key not in rating_keys):
            if section_id:
                section_ids.add(section_id)
            elif rating_key:
                rating_keys.add(rating_key)
            plexpy.CONFIG.GET_FILE_SIZES_HOLD = {'section_ids': list(section_ids), 'rating_keys': list(rating_keys)}
            library_data = libraries.Libraries()
            result = library_data.get_media_info_file_sizes(section_id=section_id, rating_key=rating_key)
            if section_id:
                section_ids.remove(section_id)
            elif rating_key:
                rating_keys.remove(rating_key)
            plexpy.CONFIG.GET_FILE_SIZES_HOLD = {'section_ids': list(section_ids), 'rating_keys': list(rating_keys)}
        else:
            result = False
        return {'success': result}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_library(self, section_id=None, include_last_accessed=False, **kwargs):
        if False:
            while True:
                i = 10
        ' Get a library\'s details.\n\n            ```\n            Required parameters:\n                section_id (str):               The id of the Plex library section\n\n            Optional parameters:\n                include_last_accessed (bool):   True to include the last_accessed value for the library.\n\n            Returns:\n                json:\n                    {"child_count": null,\n                     "count": 887,\n                     "deleted_section": 0,\n                     "do_notify": 1,\n                     "do_notify_created": 1,\n                     "is_active": 1,\n                     "keep_history": 1,\n                     "last_accessed": 1462693216,\n                     "library_art": "/:/resources/movie-fanart.jpg",\n                     "library_thumb": "/:/resources/movie.png",\n                     "parent_count": null,\n                     "row_id": 1,\n                     "section_id": 1,\n                     "section_name": "Movies",\n                     "section_type": "movie",\n                     "server_id": "ds48g4r354a8v9byrrtr697g3g79w"\n                     }\n            ```\n        '
        include_last_accessed = helpers.bool_true(include_last_accessed)
        if section_id:
            library_data = libraries.Libraries()
            library_details = library_data.get_details(section_id=section_id, include_last_accessed=include_last_accessed)
            if library_details:
                return library_details
            else:
                logger.warn('Unable to retrieve data for get_library.')
                return library_details
        else:
            logger.warn('Library details requested but no section_id received.')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_library_watch_time_stats(self, section_id=None, grouping=None, query_days=None, **kwargs):
        if False:
            print('Hello World!')
        ' Get a library\'s watch time statistics.\n\n            ```\n            Required parameters:\n                section_id (str):       The id of the Plex library section\n\n            Optional parameters:\n                grouping (int):         0 or 1\n                query_days (str):       Comma separated days, e.g. "1,7,30,0"\n\n            Returns:\n                json:\n                    [{"query_days": 1,\n                      "total_plays": 0,\n                      "total_time": 0\n                      },\n                     {"query_days": 7,\n                      "total_plays": 3,\n                      "total_time": 15694\n                      },\n                     {"query_days": 30,\n                      "total_plays": 35,\n                      "total_time": 63054\n                      },\n                     {"query_days": 0,\n                      "total_plays": 508,\n                      "total_time": 1183080\n                      }\n                     ]\n            ```\n        '
        grouping = helpers.bool_true(grouping, return_none=True)
        if section_id:
            library_data = libraries.Libraries()
            result = library_data.get_watch_time_stats(section_id=section_id, grouping=grouping, query_days=query_days)
            if result:
                return result
            else:
                logger.warn('Unable to retrieve data for get_library_watch_time_stats.')
                return result
        else:
            logger.warn('Library watch time stats requested but no section_id received.')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_library_user_stats(self, section_id=None, grouping=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Get a library\'s user statistics.\n\n            ```\n            Required parameters:\n                section_id (str):       The id of the Plex library section\n\n            Optional parameters:\n                grouping (int):         0 or 1\n\n            Returns:\n                json:\n                    [{"friendly_name": "Jon Snow",\n                      "total_plays": 170,\n                      "total_time": 349618,\n                      "user_id": 133788,\n                      "user_thumb": "https://plex.tv/users/k10w42309cynaopq/avatar",\n                      "username": "LordCommanderSnow"\n                      },\n                     {"friendly_name": "DanyKhaleesi69",\n                      "total_plays": 42,\n                      "total_time": 50185,\n                      "user_id": 8008135,\n                      "user_thumb": "https://plex.tv/users/568gwwoib5t98a3a/avatar",\n                      "username: "DanyKhaleesi69"\n                      },\n                     {...},\n                     {...}\n                     ]\n            ```\n        '
        grouping = helpers.bool_true(grouping, return_none=True)
        if section_id:
            library_data = libraries.Libraries()
            result = library_data.get_user_stats(section_id=section_id, grouping=grouping)
            if result:
                return result
            else:
                logger.warn('Unable to retrieve data for get_library_user_stats.')
                return result
        else:
            logger.warn('Library user stats requested but no section_id received.')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def delete_all_library_history(self, server_id=None, section_id=None, row_ids=None, **kwargs):
        if False:
            while True:
                i = 10
        ' Delete all Tautulli history for a specific library.\n\n            ```\n            Required parameters:\n                server_id (str):        The Plex server identifier of the library section\n                section_id (str):       The id of the Plex library section\n\n            Optional parameters:\n                row_ids (str):          Comma separated row ids to delete, e.g. "2,3,8"\n\n            Returns:\n                None\n            ```\n        '
        if server_id and section_id or row_ids:
            library_data = libraries.Libraries()
            success = library_data.delete(server_id=server_id, section_id=section_id, row_ids=row_ids, purge_only=True)
            if success:
                return {'result': 'success', 'message': 'Deleted library history.'}
            else:
                return {'result': 'error', 'message': 'Failed to delete library(s) history.'}
        else:
            return {'result': 'error', 'message': 'No server id and section id or row ids received.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def delete_library(self, server_id=None, section_id=None, row_ids=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Delete a library section from Tautulli. Also erases all history for the library.\n\n            ```\n            Required parameters:\n                server_id (str):        The Plex server identifier of the library section\n                section_id (str):       The id of the Plex library section\n\n            Optional parameters:\n                row_ids (str):          Comma separated row ids to delete, e.g. "2,3,8"\n\n            Returns:\n                None\n            ```\n        '
        if server_id and section_id or row_ids:
            library_data = libraries.Libraries()
            success = library_data.delete(server_id=server_id, section_id=section_id, row_ids=row_ids)
            if success:
                return {'result': 'success', 'message': 'Deleted library.'}
            else:
                return {'result': 'error', 'message': 'Failed to delete library(s).'}
        else:
            return {'result': 'error', 'message': 'No server id and section id or row ids received.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def undelete_library(self, section_id=None, section_name=None, **kwargs):
        if False:
            while True:
                i = 10
        ' Restore a deleted library section to Tautulli.\n\n            ```\n            Required parameters:\n                section_id (str):       The id of the Plex library section\n                section_name (str):     The name of the Plex library section\n\n            Optional parameters:\n                None\n\n            Returns:\n                None\n            ```\n        '
        library_data = libraries.Libraries()
        result = library_data.undelete(section_id=section_id, section_name=section_name)
        if result:
            if section_id:
                msg = 'section_id %s' % section_id
            elif section_name:
                msg = 'section_name %s' % section_name
            return {'result': 'success', 'message': 'Re-added library with %s.' % msg}
        return {'result': 'error', 'message': 'Unable to re-add library. Invalid section_id or section_name.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def delete_media_info_cache(self, section_id, **kwargs):
        if False:
            return 10
        ' Delete the media info table cache for a specific library.\n\n            ```\n            Required parameters:\n                section_id (str):       The id of the Plex library section\n\n            Optional parameters:\n                None\n\n            Returns:\n                None\n            ```\n        '
        get_file_sizes_hold = plexpy.CONFIG.GET_FILE_SIZES_HOLD
        section_ids = set(get_file_sizes_hold['section_ids'])
        if section_id not in section_ids:
            if section_id:
                library_data = libraries.Libraries()
                delete_row = library_data.delete_media_info_cache(section_id=section_id)
                if delete_row:
                    return {'message': delete_row}
            else:
                return {'message': 'no data received'}
        else:
            return {'message': 'Cannot delete media info cache while getting file sizes.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def delete_duplicate_libraries(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        library_data = libraries.Libraries()
        result = library_data.delete_duplicate_libraries()
        if result:
            return {'message': result}
        else:
            return {'message': 'Unable to delete duplicate libraries from the database.'}

    @cherrypy.expose
    @requireAuth()
    def users(self, **kwargs):
        if False:
            return 10
        return serve_template(template_name='users.html', title='Users')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @sanitize_out()
    @addtoapi('get_users_table')
    def get_user_list(self, grouping=None, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Get the data on Tautulli users table.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                grouping (int):                 0 or 1\n                order_column (str):             "user_thumb", "friendly_name", "last_seen", "ip_address", "platform",\n                                                "player", "last_played", "plays", "duration"\n                order_dir (str):                "desc" or "asc"\n                start (int):                    Row to start from, 0\n                length (int):                   Number of items to return, 25\n                search (str):                   A string to search for, "Jon Snow"\n\n            Returns:\n                json:\n                    {"draw": 1,\n                     "recordsTotal": 10,\n                     "recordsFiltered": 10,\n                     "data":\n                        [{"allow_guest": 1,\n                          "do_notify": 1,\n                          "duration": 2998290,\n                          "email": "Jon.Snow.1337@CastleBlack.com",\n                          "friendly_name": "Jon Snow",\n                          "guid": "com.plexapp.agents.thetvdb://121361/6/1?lang=en",\n                          "history_row_id": 1121,\n                          "ip_address": "xxx.xxx.xxx.xxx",\n                          "is_active": 1,\n                          "keep_history": 1,\n                          "last_played": "Game of Thrones - The Red Woman",\n                          "last_seen": 1462591869,\n                          "live": 0,\n                          "media_index": 1,\n                          "media_type": "episode",\n                          "originally_available_at": "2016-04-24",\n                          "parent_media_index": 6,\n                          "parent_title": "",\n                          "platform": "Chrome",\n                          "player": "Plex Web (Chrome)",\n                          "plays": 487,\n                          "rating_key": 153037,\n                          "row_id": 1,\n                          "thumb": "/library/metadata/153036/thumb/1462175062",\n                          "title": "Jon Snow",\n                          "transcode_decision": "transcode",\n                          "user_id": 133788,\n                          "user_thumb": "https://plex.tv/users/568gwwoib5t98a3a/avatar",\n                          "username": "LordCommanderSnow",\n                          "year": 2016\n                          },\n                         {...},\n                         {...}\n                         ]\n                     }\n            ```\n        '
        if not kwargs.get('json_data'):
            dt_columns = [('user_thumb', False, False), ('friendly_name', True, True), ('username', True, True), ('title', True, True), ('email', True, True), ('last_seen', True, False), ('ip_address', True, True), ('platform', True, True), ('player', True, True), ('last_played', True, False), ('plays', True, False), ('duration', True, False)]
            kwargs['json_data'] = build_datatables_json(kwargs, dt_columns, 'friendly_name')
        grouping = helpers.bool_true(grouping, return_none=True)
        user_data = users.Users()
        user_list = user_data.get_datatables_list(kwargs=kwargs, grouping=grouping)
        return user_list

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def refresh_users_list(self, **kwargs):
        if False:
            while True:
                i = 10
        ' Manually refresh the users list. '
        logger.info('Manual users list refresh requested.')
        result = users.refresh_users()
        if result:
            return {'result': 'success', 'message': 'Users list refreshed.'}
        else:
            return {'result': 'error', 'message': 'Unable to refresh users list.'}

    @cherrypy.expose
    @requireAuth()
    def user(self, user_id=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if not allow_session_user(user_id):
            raise cherrypy.HTTPRedirect(plexpy.HTTP_ROOT)
        if user_id:
            try:
                user_data = users.Users()
                user_details = user_data.get_details(user_id=user_id)
            except:
                logger.warn('Unable to retrieve user details for user_id %s ' % user_id)
                return serve_template(template_name='user.html', title='User', data=None)
        else:
            logger.debug('User page requested but no user_id received.')
            return serve_template(template_name='user.html', title='User', data=None)
        return serve_template(template_name='user.html', title='User', data=user_details)

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def edit_user_dialog(self, user=None, user_id=None, **kwargs):
        if False:
            print('Hello World!')
        if user_id:
            user_data = users.Users()
            result = user_data.get_details(user_id=user_id)
            status_message = ''
        else:
            result = None
            status_message = 'An error occured.'
        return serve_template(template_name='edit_user.html', title='Edit User', data=result, status_message=status_message)

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    @addtoapi()
    def edit_user(self, user_id=None, **kwargs):
        if False:
            print('Hello World!')
        ' Update a user on Tautulli.\n\n            ```\n            Required parameters:\n                user_id (str):              The id of the Plex user\n                friendly_name(str):         The friendly name of the user\n                custom_thumb (str):         The URL for the custom user thumbnail\n                keep_history (int):         0 or 1\n                allow_guest (int):          0 or 1\n\n            Optional paramters:\n                None\n\n            Returns:\n                None\n            ```\n        '
        friendly_name = kwargs.get('friendly_name', '')
        custom_thumb = kwargs.get('custom_thumb', '')
        do_notify = kwargs.get('do_notify', 0)
        keep_history = kwargs.get('keep_history', 0)
        allow_guest = kwargs.get('allow_guest', 0)
        if user_id:
            try:
                user_data = users.Users()
                user_data.set_config(user_id=user_id, friendly_name=friendly_name, custom_thumb=custom_thumb, do_notify=do_notify, keep_history=keep_history, allow_guest=allow_guest)
                status_message = 'Successfully updated user.'
                return status_message
            except:
                status_message = 'Failed to update user.'
                return status_message

    @cherrypy.expose
    @requireAuth()
    def user_watch_time_stats(self, user=None, user_id=None, **kwargs):
        if False:
            return 10
        if not allow_session_user(user_id):
            return serve_template(template_name='user_watch_time_stats.html', data=None, title='Watch Stats')
        if user_id or user:
            user_data = users.Users()
            result = user_data.get_watch_time_stats(user_id=user_id)
        else:
            result = None
        if result:
            return serve_template(template_name='user_watch_time_stats.html', data=result, title='Watch Stats')
        else:
            logger.warn('Unable to retrieve data for user_watch_time_stats.')
            return serve_template(template_name='user_watch_time_stats.html', data=None, title='Watch Stats')

    @cherrypy.expose
    @requireAuth()
    def user_player_stats(self, user=None, user_id=None, **kwargs):
        if False:
            while True:
                i = 10
        if not allow_session_user(user_id):
            return serve_template(template_name='user_player_stats.html', data=None, title='Player Stats')
        if user_id or user:
            user_data = users.Users()
            result = user_data.get_player_stats(user_id=user_id)
        else:
            result = None
        if result:
            return serve_template(template_name='user_player_stats.html', data=result, title='Player Stats')
        else:
            logger.warn('Unable to retrieve data for user_player_stats.')
            return serve_template(template_name='user_player_stats.html', data=None, title='Player Stats')

    @cherrypy.expose
    @requireAuth()
    def get_user_recently_watched(self, user=None, user_id=None, limit='10', **kwargs):
        if False:
            while True:
                i = 10
        if not allow_session_user(user_id):
            return serve_template(template_name='user_recently_watched.html', data=None, title='Recently Watched')
        if user_id or user:
            user_data = users.Users()
            result = user_data.get_recently_watched(user_id=user_id, limit=limit)
        else:
            result = None
        if result:
            return serve_template(template_name='user_recently_watched.html', data=result, title='Recently Watched')
        else:
            logger.warn('Unable to retrieve data for get_user_recently_watched.')
            return serve_template(template_name='user_recently_watched.html', data=None, title='Recently Watched')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @sanitize_out()
    @addtoapi()
    def get_user_ips(self, user_id=None, **kwargs):
        if False:
            while True:
                i = 10
        ' Get the data on Tautulli users IP table.\n\n            ```\n            Required parameters:\n                user_id (str):                  The id of the Plex user\n\n            Optional parameters:\n                order_column (str):             "last_seen", "first_seen", "ip_address", "platform",\n                                                "player", "last_played", "play_count"\n                order_dir (str):                "desc" or "asc"\n                start (int):                    Row to start from, 0\n                length (int):                   Number of items to return, 25\n                search (str):                   A string to search for, "xxx.xxx.xxx.xxx"\n\n            Returns:\n                json:\n                    {"draw": 1,\n                     "recordsTotal": 2344,\n                     "recordsFiltered": 10,\n                     "data":\n                        [{"friendly_name": "Jon Snow",\n                          "guid": "com.plexapp.agents.thetvdb://121361/6/1?lang=en",\n                          "id": 1121,\n                          "ip_address": "xxx.xxx.xxx.xxx",\n                          "last_played": "Game of Thrones - The Red Woman",\n                          "last_seen": 1462591869,\n                          "first_seen": 1583968210,\n                          "live": 0,\n                          "media_index": 1,\n                          "media_type": "episode",\n                          "originally_available_at": "2016-04-24",\n                          "parent_media_index": 6,\n                          "parent_title": "",\n                          "platform": "Chrome",\n                          "play_count": 149,\n                          "player": "Plex Web (Chrome)",\n                          "rating_key": 153037,\n                          "thumb": "/library/metadata/153036/thumb/1462175062",\n                          "transcode_decision": "transcode",\n                          "user_id": 133788,\n                          "year": 2016\n                          },\n                         {...},\n                         {...}\n                         ]\n                     }\n            ```\n        '
        if not kwargs.get('json_data'):
            dt_columns = [('last_seen', True, False), ('first_seen', True, False), ('ip_address', True, True), ('platform', True, True), ('player', True, True), ('last_played', True, True), ('play_count', True, True)]
            kwargs['json_data'] = build_datatables_json(kwargs, dt_columns, 'last_seen')
        user_data = users.Users()
        history = user_data.get_datatables_unique_ips(user_id=user_id, kwargs=kwargs)
        return history

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @sanitize_out()
    @addtoapi()
    def get_user_logins(self, user_id=None, **kwargs):
        if False:
            while True:
                i = 10
        ' Get the data on Tautulli user login table.\n\n            ```\n            Required parameters:\n                user_id (str):                  The id of the Plex user\n\n            Optional parameters:\n                order_column (str):             "date", "time", "ip_address", "host", "os", "browser"\n                order_dir (str):                "desc" or "asc"\n                start (int):                    Row to start from, 0\n                length (int):                   Number of items to return, 25\n                search (str):                   A string to search for, "xxx.xxx.xxx.xxx"\n\n            Returns:\n                json:\n                    {"draw": 1,\n                     "recordsTotal": 2344,\n                     "recordsFiltered": 10,\n                     "data":\n                        [{"browser": "Safari 7.0.3",\n                          "current": false,\n                          "expiry": "2021-06-30 18:48:03",\n                          "friendly_name": "Jon Snow",\n                          "host": "http://plexpy.castleblack.com",\n                          "ip_address": "xxx.xxx.xxx.xxx",\n                          "os": "Mac OS X",\n                          "row_id": 1,\n                          "timestamp": 1462591869,\n                          "user": "LordCommanderSnow",\n                          "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/7046A194A",\n                          "user_group": "guest",\n                          "user_id": 133788\n                          },\n                         {...},\n                         {...}\n                         ]\n                     }\n            ```\n        '
        if not kwargs.get('json_data'):
            dt_columns = [('timestamp', True, False), ('ip_address', True, True), ('host', True, True), ('os', True, True), ('browser', True, True)]
            kwargs['json_data'] = build_datatables_json(kwargs, dt_columns, 'timestamp')
        jwt_token = get_jwt_token()
        user_data = users.Users()
        history = user_data.get_datatables_user_login(user_id=user_id, jwt_token=jwt_token, kwargs=kwargs)
        return history

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def logout_user_session(self, row_ids=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Logout Tautulli user sessions.\n\n            ```\n            Required parameters:\n                row_ids (str):          Comma separated row ids to sign out, e.g. "2,3,8"\n\n            Optional parameters:\n                None\n\n            Returns:\n                None\n            ```\n        '
        user_data = users.Users()
        result = user_data.clear_user_login_token(row_ids=row_ids)
        if result:
            return {'result': 'success', 'message': 'Users session logged out.'}
        else:
            return {'result': 'error', 'message': 'Unable to logout user session.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_user(self, user_id=None, include_last_seen=False, **kwargs):
        if False:
            print('Hello World!')
        ' Get a user\'s details.\n\n            ```\n            Required parameters:\n                user_id (str):              The id of the Plex user\n\n            Optional parameters:\n                include_last_seen (bool):   True to include the last_seen value for the user.\n\n            Returns:\n                json:\n                    {"allow_guest": 1,\n                     "deleted_user": 0,\n                     "do_notify": 1,\n                     "email": "Jon.Snow.1337@CastleBlack.com",\n                     "friendly_name": "Jon Snow",\n                     "is_active": 1,\n                     "is_admin": 0,\n                     "is_allow_sync": 1,\n                     "is_home_user": 1,\n                     "is_restricted": 0,\n                     "keep_history": 1,\n                     "last_seen": 1462591869,\n                     "row_id": 1,\n                     "shared_libraries": ["10", "1", "4", "5", "15", "20", "2"],\n                     "user_id": 133788,\n                     "user_thumb": "https://plex.tv/users/k10w42309cynaopq/avatar",\n                     "username": "LordCommanderSnow"\n                     }\n            ```\n        '
        include_last_seen = helpers.bool_true(include_last_seen)
        if user_id:
            user_data = users.Users()
            user_details = user_data.get_details(user_id=user_id, include_last_seen=include_last_seen)
            if user_details:
                return user_details
            else:
                logger.warn('Unable to retrieve data for get_user.')
                return user_details
        else:
            logger.warn('User details requested but no user_id received.')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_user_watch_time_stats(self, user_id=None, grouping=None, query_days=None, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Get a user\'s watch time statistics.\n\n            ```\n            Required parameters:\n                user_id (str):          The id of the Plex user\n\n            Optional parameters:\n                grouping (int):         0 or 1\n                query_days (str):       Comma separated days, e.g. "1,7,30,0"\n\n            Returns:\n                json:\n                    [{"query_days": 1,\n                      "total_plays": 0,\n                      "total_time": 0\n                      },\n                     {"query_days": 7,\n                      "total_plays": 3,\n                      "total_time": 15694\n                      },\n                     {"query_days": 30,\n                      "total_plays": 35,\n                      "total_time": 63054\n                      },\n                     {"query_days": 0,\n                      "total_plays": 508,\n                      "total_time": 1183080\n                      }\n                     ]\n            ```\n        '
        grouping = helpers.bool_true(grouping, return_none=True)
        if user_id:
            user_data = users.Users()
            result = user_data.get_watch_time_stats(user_id=user_id, grouping=grouping, query_days=query_days)
            if result:
                return result
            else:
                logger.warn('Unable to retrieve data for get_user_watch_time_stats.')
                return result
        else:
            logger.warn('User watch time stats requested but no user_id received.')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_user_player_stats(self, user_id=None, grouping=None, **kwargs):
        if False:
            print('Hello World!')
        ' Get a user\'s player statistics.\n\n            ```\n            Required parameters:\n                user_id (str):          The id of the Plex user\n\n            Optional parameters:\n                grouping (int):         0 or 1\n\n            Returns:\n                json:\n                    [{"platform": "Chrome",\n                      "platform_name": "chrome",\n                      "player_name": "Plex Web (Chrome)",\n                      "result_id": 1,\n                      "total_plays": 170,\n                      "total_time": 349618\n                      },\n                     {"platform": "Chromecast",\n                      "platform_name": "chromecast",\n                      "player_name": "Chromecast",\n                      "result_id": 2,\n                      "total_plays": 42,\n                      "total_time": 50185\n                      },\n                     {...},\n                     {...}\n                     ]\n            ```\n        '
        grouping = helpers.bool_true(grouping, return_none=True)
        if user_id:
            user_data = users.Users()
            result = user_data.get_player_stats(user_id=user_id, grouping=grouping)
            if result:
                return result
            else:
                logger.warn('Unable to retrieve data for get_user_player_stats.')
                return result
        else:
            logger.warn('User watch time stats requested but no user_id received.')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def delete_all_user_history(self, user_id=None, row_ids=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Delete all Tautulli history for a specific user.\n\n            ```\n            Required parameters:\n                user_id (str):          The id of the Plex user\n\n            Optional parameters:\n                row_ids (str):          Comma separated row ids to delete, e.g. "2,3,8"\n\n            Returns:\n                None\n            ```\n        '
        if user_id or row_ids:
            user_data = users.Users()
            success = user_data.delete(user_id=user_id, row_ids=row_ids, purge_only=True)
            if success:
                return {'result': 'success', 'message': 'Deleted user history.'}
            else:
                return {'result': 'error', 'message': 'Failed to delete user(s) history.'}
        else:
            return {'result': 'error', 'message': 'No user id or row ids received.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def delete_user(self, user_id=None, row_ids=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Delete a user from Tautulli. Also erases all history for the user.\n\n            ```\n            Required parameters:\n                user_id (str):          The id of the Plex user\n\n            Optional parameters:\n                row_ids (str):          Comma separated row ids to delete, e.g. "2,3,8"\n\n            Returns:\n                None\n            ```\n        '
        if user_id or row_ids:
            user_data = users.Users()
            success = user_data.delete(user_id=user_id, row_ids=row_ids)
            if success:
                return {'result': 'success', 'message': 'Deleted user.'}
            else:
                return {'result': 'error', 'message': 'Failed to delete user(s).'}
        else:
            return {'result': 'error', 'message': 'No user id or row ids received.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def undelete_user(self, user_id=None, username=None, **kwargs):
        if False:
            print('Hello World!')
        ' Restore a deleted user to Tautulli.\n\n            ```\n            Required parameters:\n                user_id (str):          The id of the Plex user\n                username (str):         The username of the Plex user\n\n            Optional parameters:\n                None\n\n            Returns:\n                None\n            ```\n        '
        user_data = users.Users()
        result = user_data.undelete(user_id=user_id, username=username)
        if result:
            if user_id:
                msg = 'user_id %s' % user_id
            elif username:
                msg = 'username %s' % username
            return {'result': 'success', 'message': 'Re-added user with %s.' % msg}
        return {'result': 'error', 'message': 'Unable to re-add user. Invalid user_id or username.'}

    @cherrypy.expose
    @requireAuth()
    def history(self, **kwargs):
        if False:
            while True:
                i = 10
        config = {'database_is_importing': database.IS_IMPORTING}
        return serve_template(template_name='history.html', title='History', config=config)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @sanitize_out()
    @addtoapi()
    def get_history(self, user=None, user_id=None, grouping=None, include_activity=None, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Get the Tautulli history.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                grouping (int):                 0 or 1\n                include_activity (int):         0 or 1\n                user (str):                     "Jon Snow"\n                user_id (int):                  133788\n                rating_key (int):               4348\n                parent_rating_key (int):        544\n                grandparent_rating_key (int):   351\n                start_date (str):               History for the exact date, "YYYY-MM-DD"\n                before (str):                   History before and including the date, "YYYY-MM-DD"\n                after (str):                    History after and including the date, "YYYY-MM-DD"\n                section_id (int):               2\n                media_type (str):               "movie", "episode", "track", "live", "collection", "playlist"\n                transcode_decision (str):       "direct play", "copy", "transcode",\n                guid (str):                     Plex guid for an item, e.g. "com.plexapp.agents.thetvdb://121361/6/1"\n                order_column (str):             "date", "friendly_name", "ip_address", "platform", "player",\n                                                "full_title", "started", "paused_counter", "stopped", "duration"\n                order_dir (str):                "desc" or "asc"\n                start (int):                    Row to start from, 0\n                length (int):                   Number of items to return, 25\n                search (str):                   A string to search for, "Thrones"\n\n            Returns:\n                json:\n                    {"draw": 1,\n                     "recordsTotal": 1000,\n                     "recordsFiltered": 250,\n                     "total_duration": "42 days 5 hrs 18 mins",\n                     "filter_duration": "10 hrs 12 mins",\n                     "data":\n                        [{"date": 1462687607,\n                          "friendly_name": "Mother of Dragons",\n                          "full_title": "Game of Thrones - The Red Woman",\n                          "grandparent_rating_key": 351,\n                          "grandparent_title": "Game of Thrones",\n                          "original_title": "",\n                          "group_count": 1,\n                          "group_ids": "1124",\n                          "guid": "com.plexapp.agents.thetvdb://121361/6/1?lang=en",\n                          "ip_address": "xxx.xxx.xxx.xxx",\n                          "live": 0,\n                          "location": "wan",\n                          "machine_id": "lmd93nkn12k29j2lnm",\n                          "media_index": 17,\n                          "media_type": "episode",\n                          "originally_available_at": "2016-04-24",\n                          "parent_media_index": 7,\n                          "parent_rating_key": 544,\n                          "parent_title": "",\n                          "paused_counter": 0,\n                          "percent_complete": 84,\n                          "platform": "Windows",\n                          "play_duration": 263,\n                          "product": "Plex for Windows",\n                          "player": "Castle-PC",\n                          "rating_key": 4348,\n                          "reference_id": 1123,\n                          "relayed": 0,\n                          "row_id": 1124,\n                          "secure": 1,\n                          "session_key": null,\n                          "started": 1462688107,\n                          "state": null,\n                          "stopped": 1462688370,\n                          "thumb": "/library/metadata/4348/thumb/1462414561",\n                          "title": "The Red Woman",\n                          "transcode_decision": "transcode",\n                          "user": "DanyKhaleesi69",\n                          "user_id": 8008135,\n                          "watched_status": 0,\n                          "year": 2016\n                          },\n                         {...},\n                         {...}\n                         ]\n                     }\n            ```\n        '
        if not kwargs.get('json_data'):
            dt_columns = [('date', True, False), ('friendly_name', True, True), ('ip_address', True, True), ('platform', True, True), ('product', True, True), ('player', True, True), ('full_title', True, True), ('started', True, False), ('paused_counter', True, False), ('stopped', True, False), ('duration', True, False), ('watched_status', False, False)]
            kwargs['json_data'] = build_datatables_json(kwargs, dt_columns, 'date')
        grouping = helpers.bool_true(grouping, return_none=True)
        include_activity = helpers.bool_true(include_activity, return_none=True)
        custom_where = []
        if user_id:
            user_id = helpers.split_strip(user_id)
            if user_id:
                custom_where.append(['session_history.user_id', user_id])
        elif user:
            user = helpers.split_strip(user)
            if user:
                custom_where.append(['session_history.user', user])
        if 'rating_key' in kwargs:
            if kwargs.get('media_type') in ('collection', 'playlist') and kwargs.get('rating_key'):
                pms_connect = pmsconnect.PmsConnect()
                result = pms_connect.get_item_children(rating_key=kwargs.pop('rating_key'), media_type=kwargs.pop('media_type'))
                rating_keys = [child['rating_key'] for child in result['children_list']]
                custom_where.append(['session_history_metadata.rating_key OR', rating_keys])
                custom_where.append(['session_history_metadata.parent_rating_key OR', rating_keys])
                custom_where.append(['session_history_metadata.grandparent_rating_key OR', rating_keys])
            else:
                rating_key = helpers.split_strip(kwargs.pop('rating_key', ''))
                if rating_key:
                    custom_where.append(['session_history.rating_key', rating_key])
        if 'parent_rating_key' in kwargs:
            rating_key = helpers.split_strip(kwargs.pop('parent_rating_key', ''))
            if rating_key:
                custom_where.append(['session_history.parent_rating_key', rating_key])
        if 'grandparent_rating_key' in kwargs:
            rating_key = helpers.split_strip(kwargs.pop('grandparent_rating_key', ''))
            if rating_key:
                custom_where.append(['session_history.grandparent_rating_key', rating_key])
        if 'start_date' in kwargs:
            start_date = helpers.split_strip(kwargs.pop('start_date', ''))
            if start_date:
                custom_where.append(['strftime("%Y-%m-%d", datetime(started, "unixepoch", "localtime"))', start_date])
        if 'before' in kwargs:
            before = helpers.split_strip(kwargs.pop('before', ''))
            if before:
                custom_where.append(['strftime("%Y-%m-%d", datetime(started, "unixepoch", "localtime")) <', before])
        if 'after' in kwargs:
            after = helpers.split_strip(kwargs.pop('after', ''))
            if after:
                custom_where.append(['strftime("%Y-%m-%d", datetime(started, "unixepoch", "localtime")) >', after])
        if 'reference_id' in kwargs:
            reference_id = helpers.split_strip(kwargs.pop('reference_id', ''))
            if reference_id:
                custom_where.append(['session_history.reference_id', reference_id])
        if 'section_id' in kwargs:
            section_id = helpers.split_strip(kwargs.pop('section_id', ''))
            if section_id:
                custom_where.append(['session_history.section_id', section_id])
        if 'media_type' in kwargs:
            media_type = helpers.split_strip(kwargs.pop('media_type', ''))
            if media_type and 'all' not in media_type:
                custom_where.append(['media_type_live', media_type])
        if 'transcode_decision' in kwargs:
            transcode_decision = helpers.split_strip(kwargs.pop('transcode_decision', ''))
            if transcode_decision and 'all' not in transcode_decision:
                custom_where.append(['session_history_media_info.transcode_decision', transcode_decision])
        if 'guid' in kwargs:
            guid = helpers.split_strip(kwargs.pop('guid', '').split('?')[0])
            if guid:
                custom_where.append(['session_history_metadata.guid', ['LIKE ' + g + '%' for g in guid]])
        data_factory = datafactory.DataFactory()
        history = data_factory.get_datatables_history(kwargs=kwargs, custom_where=custom_where, grouping=grouping, include_activity=include_activity)
        return history

    @cherrypy.expose
    @requireAuth()
    def get_stream_data(self, row_id=None, session_key=None, user=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        data_factory = datafactory.DataFactory()
        stream_data = data_factory.get_stream_details(row_id, session_key)
        return serve_template(template_name='stream_data.html', title='Stream Data', data=stream_data, user=user)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @addtoapi('get_stream_data')
    def get_stream_data_api(self, row_id=None, session_key=None, **kwargs):
        if False:
            while True:
                i = 10
        ' Get the stream details from history or current stream.\n\n            ```\n            Required parameters:\n                row_id (int):       The row ID number for the history item, OR\n                session_key (int):  The session key of the current stream\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    {"aspect_ratio": "2.35",\n                     "audio_bitrate": 231,\n                     "audio_channels": 6,\n                     "audio_language": "English",\n                     "audio_language_code": "eng",\n                     "audio_codec": "aac",\n                     "audio_decision": "transcode",\n                     "bitrate": 2731,\n                     "container": "mp4",\n                     "current_session": "",\n                     "grandparent_title": "",\n                     "media_type": "movie",\n                     "optimized_version": "",\n                     "optimized_version_profile": "",\n                     "optimized_version_title": "",\n                     "original_title": "",\n                     "pre_tautulli": "",\n                     "quality_profile": "1.5 Mbps 480p",\n                     "stream_audio_bitrate": 203,\n                     "stream_audio_channels": 2,\n                     "stream_audio_language": "English",\n                     "stream_audio_language_code", "eng",\n                     "stream_audio_codec": "aac",\n                     "stream_audio_decision": "transcode",\n                     "stream_bitrate": 730,\n                     "stream_container": "mkv",\n                     "stream_container_decision": "transcode",\n                     "stream_subtitle_codec": "",\n                     "stream_subtitle_decision": "",\n                     "stream_video_bitrate": 527,\n                     "stream_video_codec": "h264",\n                     "stream_video_decision": "transcode",\n                     "stream_video_dynamic_range": "SDR",\n                     "stream_video_framerate": "24p",\n                     "stream_video_height": 306,\n                     "stream_video_resolution": "SD",\n                     "stream_video_width": 720,\n                     "subtitle_codec": "",\n                     "subtitles": "",\n                     "synced_version": "",\n                     "synced_version_profile": "",\n                     "title": "Frozen",\n                     "transcode_hw_decoding": "",\n                     "transcode_hw_encoding": "",\n                     "video_bitrate": 2500,\n                     "video_codec": "h264",\n                     "video_decision": "transcode",\n                     "video_dynamic_range": "SDR",\n                     "video_framerate": "24p",\n                     "video_height": 816,\n                     "video_resolution": "1080",\n                     "video_width": 1920\n                     }\n            ```\n        '
        if 'id' in kwargs:
            row_id = kwargs['id']
        data_factory = datafactory.DataFactory()
        stream_data = data_factory.get_stream_details(row_id, session_key)
        return stream_data

    @cherrypy.expose
    @requireAuth()
    def get_ip_address_details(self, ip_address=None, **kwargs):
        if False:
            return 10
        if not helpers.is_valid_ip(ip_address):
            ip_address = None
        public = helpers.is_public_ip(ip_address)
        return serve_template(template_name='ip_address_modal.html', title='IP Address Details', data=ip_address, public=public, kwargs=kwargs)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi('delete_history')
    def delete_history_rows(self, row_ids=None, **kwargs):
        if False:
            print('Hello World!')
        ' Delete history rows from Tautulli.\n\n            ```\n            Required parameters:\n                row_ids (str):          Comma separated row ids to delete, e.g. "65,110,2,3645"\n\n            Optional parameters:\n                None\n\n            Returns:\n                None\n            ```\n        '
        data_factory = datafactory.DataFactory()
        if row_ids:
            success = database.delete_session_history_rows(row_ids=row_ids)
            if success:
                return {'result': 'success', 'message': 'Deleted history.'}
            else:
                return {'result': 'error', 'message': 'Failed to delete history.'}
        else:
            return {'result': 'error', 'message': 'No row ids received.'}

    @cherrypy.expose
    @requireAuth()
    def graphs(self, **kwargs):
        if False:
            while True:
                i = 10
        return serve_template(template_name='graphs.html', title='Graphs')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @sanitize_out()
    @addtoapi()
    def get_user_names(self, **kwargs):
        if False:
            return 10
        ' Get a list of all user and user ids.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    [{"friendly_name": "Jon Snow", "user_id": 133788},\n                     {"friendly_name": "DanyKhaleesi69", "user_id": 8008135},\n                     {"friendly_name": "Tyrion Lannister", "user_id": 696969},\n                     {...},\n                    ]\n            ```\n        '
        user_data = users.Users()
        user_names = user_data.get_user_names(kwargs=kwargs)
        return user_names

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @addtoapi()
    def get_plays_by_date(self, time_range='30', user_id=None, y_axis='plays', grouping=None, **kwargs):
        if False:
            while True:
                i = 10
        ' Get graph data by date.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                time_range (str):       The number of days of data to return\n                y_axis (str):           "plays" or "duration"\n                user_id (str):          Comma separated list of user id to filter the data\n                grouping (int):         0 or 1\n\n            Returns:\n                json:\n                    {"categories":\n                        ["YYYY-MM-DD", "YYYY-MM-DD", ...]\n                     "series":\n                        [{"name": "Movies", "data": [...]}\n                         {"name": "TV", "data": [...]},\n                         {"name": "Music", "data": [...]},\n                         {"name": "Live TV", "data": [...]}\n                         ]\n                     }\n            ```\n        '
        grouping = helpers.bool_true(grouping, return_none=True)
        graph = graphs.Graphs()
        result = graph.get_total_plays_per_day(time_range=time_range, y_axis=y_axis, user_id=user_id, grouping=grouping)
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_plays_by_date.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @addtoapi()
    def get_plays_by_dayofweek(self, time_range='30', user_id=None, y_axis='plays', grouping=None, **kwargs):
        if False:
            while True:
                i = 10
        ' Get graph data by day of the week.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                time_range (str):       The number of days of data to return\n                y_axis (str):           "plays" or "duration"\n                user_id (str):          Comma separated list of user id to filter the data\n                grouping (int):         0 or 1\n\n            Returns:\n                json:\n                    {"categories":\n                        ["Sunday", "Monday", "Tuesday", ..., "Saturday"]\n                     "series":\n                        [{"name": "Movies", "data": [...]}\n                         {"name": "TV", "data": [...]},\n                         {"name": "Music", "data": [...]},\n                         {"name": "Live TV", "data": [...]}\n                         ]\n                     }\n            ```\n        '
        grouping = helpers.bool_true(grouping, return_none=True)
        graph = graphs.Graphs()
        result = graph.get_total_plays_per_dayofweek(time_range=time_range, y_axis=y_axis, user_id=user_id, grouping=grouping)
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_plays_by_dayofweek.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @addtoapi()
    def get_plays_by_hourofday(self, time_range='30', user_id=None, y_axis='plays', grouping=None, **kwargs):
        if False:
            print('Hello World!')
        ' Get graph data by hour of the day.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                time_range (str):       The number of days of data to return\n                y_axis (str):           "plays" or "duration"\n                user_id (str):          Comma separated list of user id to filter the data\n                grouping (int):         0 or 1\n\n            Returns:\n                json:\n                    {"categories":\n                        ["00", "01", "02", ..., "23"]\n                     "series":\n                        [{"name": "Movies", "data": [...]}\n                         {"name": "TV", "data": [...]},\n                         {"name": "Music", "data": [...]},\n                         {"name": "Live TV", "data": [...]}\n                         ]\n                     }\n            ```\n        '
        grouping = helpers.bool_true(grouping, return_none=True)
        graph = graphs.Graphs()
        result = graph.get_total_plays_per_hourofday(time_range=time_range, y_axis=y_axis, user_id=user_id, grouping=grouping)
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_plays_by_hourofday.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @addtoapi()
    def get_plays_per_month(self, time_range='12', y_axis='plays', user_id=None, grouping=None, **kwargs):
        if False:
            print('Hello World!')
        ' Get graph data by month.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                time_range (str):       The number of months of data to return\n                y_axis (str):           "plays" or "duration"\n                user_id (str):          Comma separated list of user id to filter the data\n                grouping (int):         0 or 1\n\n            Returns:\n                json:\n                    {"categories":\n                        ["Jan 2016", "Feb 2016", "Mar 2016", ...]\n                     "series":\n                        [{"name": "Movies", "data": [...]}\n                         {"name": "TV", "data": [...]},\n                         {"name": "Music", "data": [...]},\n                         {"name": "Live TV", "data": [...]}\n                         ]\n                     }\n            ```\n        '
        grouping = helpers.bool_true(grouping, return_none=True)
        graph = graphs.Graphs()
        result = graph.get_total_plays_per_month(time_range=time_range, y_axis=y_axis, user_id=user_id, grouping=grouping)
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_plays_per_month.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @addtoapi()
    def get_plays_by_top_10_platforms(self, time_range='30', y_axis='plays', user_id=None, grouping=None, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Get graph data by top 10 platforms.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                time_range (str):       The number of days of data to return\n                y_axis (str):           "plays" or "duration"\n                user_id (str):          Comma separated list of user id to filter the data\n                grouping (int):         0 or 1\n\n            Returns:\n                json:\n                    {"categories":\n                        ["iOS", "Android", "Chrome", ...]\n                     "series":\n                        [{"name": "Movies", "data": [...]}\n                         {"name": "TV", "data": [...]},\n                         {"name": "Music", "data": [...]},\n                         {"name": "Live TV", "data": [...]}\n                         ]\n                     }\n            ```\n        '
        grouping = helpers.bool_true(grouping, return_none=True)
        graph = graphs.Graphs()
        result = graph.get_total_plays_by_top_10_platforms(time_range=time_range, y_axis=y_axis, user_id=user_id, grouping=grouping)
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_plays_by_top_10_platforms.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @addtoapi()
    def get_plays_by_top_10_users(self, time_range='30', y_axis='plays', user_id=None, grouping=None, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Get graph data by top 10 users.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                time_range (str):       The number of days of data to return\n                y_axis (str):           "plays" or "duration"\n                user_id (str):          Comma separated list of user id to filter the data\n                grouping (int):         0 or 1\n\n            Returns:\n                json:\n                    {"categories":\n                        ["Jon Snow", "DanyKhaleesi69", "A Girl", ...]\n                     "series":\n                        [{"name": "Movies", "data": [...]}\n                         {"name": "TV", "data": [...]},\n                         {"name": "Music", "data": [...]},\n                         {"name": "Live TV", "data": [...]}\n                         ]\n                     }\n            ```\n        '
        grouping = helpers.bool_true(grouping, return_none=True)
        graph = graphs.Graphs()
        result = graph.get_total_plays_by_top_10_users(time_range=time_range, y_axis=y_axis, user_id=user_id, grouping=grouping)
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_plays_by_top_10_users.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @addtoapi()
    def get_plays_by_stream_type(self, time_range='30', y_axis='plays', user_id=None, grouping=None, **kwargs):
        if False:
            return 10
        ' Get graph data by stream type by date.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                time_range (str):       The number of days of data to return\n                y_axis (str):           "plays" or "duration"\n                user_id (str):          Comma separated list of user id to filter the data\n                grouping (int):         0 or 1\n\n            Returns:\n                json:\n                    {"categories":\n                        ["YYYY-MM-DD", "YYYY-MM-DD", ...]\n                     "series":\n                        [{"name": "Direct Play", "data": [...]}\n                         {"name": "Direct Stream", "data": [...]},\n                         {"name": "Transcode", "data": [...]}\n                         ]\n                     }\n            ```\n        '
        grouping = helpers.bool_true(grouping, return_none=True)
        graph = graphs.Graphs()
        result = graph.get_total_plays_per_stream_type(time_range=time_range, y_axis=y_axis, user_id=user_id, grouping=grouping)
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_plays_by_stream_type.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @addtoapi()
    def get_concurrent_streams_by_stream_type(self, time_range='30', user_id=None, **kwargs):
        if False:
            print('Hello World!')
        ' Get graph data for concurrent streams by stream type by date.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                time_range (str):       The number of days of data to return\n                user_id (str):          Comma separated list of user id to filter the data\n\n            Returns:\n                json:\n                    {"categories":\n                        ["YYYY-MM-DD", "YYYY-MM-DD", ...]\n                     "series":\n                        [{"name": "Direct Play", "data": [...]}\n                         {"name": "Direct Stream", "data": [...]},\n                         {"name": "Transcode", "data": [...]},\n                         {"name": "Max. Concurrent Streams", "data":  [...]}\n                         ]\n                     }\n            ```\n        '
        graph = graphs.Graphs()
        result = graph.get_total_concurrent_streams_per_stream_type(time_range=time_range, user_id=user_id)
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_concurrent_streams_by_stream_type.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @addtoapi()
    def get_plays_by_source_resolution(self, time_range='30', y_axis='plays', user_id=None, grouping=None, **kwargs):
        if False:
            print('Hello World!')
        ' Get graph data by source resolution.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                time_range (str):       The number of days of data to return\n                y_axis (str):           "plays" or "duration"\n                user_id (str):          Comma separated list of user id to filter the data\n                grouping (int):         0 or 1\n\n            Returns:\n                json:\n                    {"categories":\n                        ["720", "1080", "sd", ...]\n                     "series":\n                        [{"name": "Direct Play", "data": [...]}\n                         {"name": "Direct Stream", "data": [...]},\n                         {"name": "Transcode", "data": [...]}\n                         ]\n                     }\n            ```\n        '
        grouping = helpers.bool_true(grouping, return_none=True)
        graph = graphs.Graphs()
        result = graph.get_total_plays_by_source_resolution(time_range=time_range, y_axis=y_axis, user_id=user_id, grouping=grouping)
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_plays_by_source_resolution.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @addtoapi()
    def get_plays_by_stream_resolution(self, time_range='30', y_axis='plays', user_id=None, grouping=None, **kwargs):
        if False:
            while True:
                i = 10
        ' Get graph data by stream resolution.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                time_range (str):       The number of days of data to return\n                y_axis (str):           "plays" or "duration"\n                user_id (str):          Comma separated list of user id to filter the data\n                grouping (int):         0 or 1\n\n            Returns:\n                json:\n                    {"categories":\n                        ["720", "1080", "sd", ...]\n                     "series":\n                        [{"name": "Direct Play", "data": [...]}\n                         {"name": "Direct Stream", "data": [...]},\n                         {"name": "Transcode", "data": [...]}\n                         ]\n                     }\n            ```\n        '
        grouping = helpers.bool_true(grouping, return_none=True)
        graph = graphs.Graphs()
        result = graph.get_total_plays_by_stream_resolution(time_range=time_range, y_axis=y_axis, user_id=user_id, grouping=grouping)
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_plays_by_stream_resolution.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @addtoapi()
    def get_stream_type_by_top_10_users(self, time_range='30', y_axis='plays', user_id=None, grouping=None, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Get graph data by stream type by top 10 users.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                time_range (str):       The number of days of data to return\n                y_axis (str):           "plays" or "duration"\n                user_id (str):          Comma separated list of user id to filter the data\n                grouping (int):         0 or 1\n\n            Returns:\n                json:\n                    {"categories":\n                        ["Jon Snow", "DanyKhaleesi69", "A Girl", ...]\n                     "series":\n                        [{"name": "Direct Play", "data": [...]}\n                         {"name": "Direct Stream", "data": [...]},\n                         {"name": "Transcode", "data": [...]}\n                        ]\n                     }\n            ```\n        '
        grouping = helpers.bool_true(grouping, return_none=True)
        graph = graphs.Graphs()
        result = graph.get_stream_type_by_top_10_users(time_range=time_range, y_axis=y_axis, user_id=user_id, grouping=grouping)
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_stream_type_by_top_10_users.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @addtoapi()
    def get_stream_type_by_top_10_platforms(self, time_range='30', y_axis='plays', user_id=None, grouping=None, **kwargs):
        if False:
            print('Hello World!')
        ' Get graph data by stream type by top 10 platforms.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                time_range (str):       The number of days of data to return\n                y_axis (str):           "plays" or "duration"\n                user_id (str):          Comma separated list of user id to filter the data\n                grouping (int):         0 or 1\n\n            Returns:\n                json:\n                    {"categories":\n                        ["iOS", "Android", "Chrome", ...]\n                     "series":\n                        [{"name": "Direct Play", "data": [...]}\n                         {"name": "Direct Stream", "data": [...]},\n                         {"name": "Transcode", "data": [...]}\n                         ]\n                     }\n            ```\n        '
        grouping = helpers.bool_true(grouping, return_none=True)
        graph = graphs.Graphs()
        result = graph.get_stream_type_by_top_10_platforms(time_range=time_range, y_axis=y_axis, user_id=user_id, grouping=grouping)
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_stream_type_by_top_10_platforms.')
            return result

    @cherrypy.expose
    @requireAuth()
    def history_table_modal(self, **kwargs):
        if False:
            return 10
        if kwargs.get('user_id') and (not allow_session_user(kwargs['user_id'])):
            return serve_template(template_name='history_table_modal.html', title='History Data', data=None)
        return serve_template(template_name='history_table_modal.html', title='History Data', data=kwargs)

    @cherrypy.expose
    @requireAuth()
    def sync(self, **kwargs):
        if False:
            i = 10
            return i + 15
        return serve_template(template_name='sync.html', title='Synced Items')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @sanitize_out()
    @requireAuth()
    def get_sync(self, machine_id=None, user_id=None, **kwargs):
        if False:
            print('Hello World!')
        if user_id == 'null':
            user_id = None
        if get_session_user_id():
            user_id = get_session_user_id()
        plex_tv = plextv.PlexTV(token=plexpy.CONFIG.PMS_TOKEN)
        result = plex_tv.get_synced_items(machine_id=machine_id, user_id_filter=user_id)
        if result:
            output = {'data': result}
        else:
            logger.warn('Unable to retrieve data for get_sync.')
            output = {'data': []}
        return output

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi('delete_synced_item')
    def delete_sync_rows(self, client_id=None, sync_id=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Delete a synced item from a device.\n\n            ```\n            Required parameters:\n                client_id (str):        The client ID of the device to delete from\n                sync_id (str):          The sync ID of the synced item\n\n            Optional parameters:\n                None\n\n            Returns:\n                None\n            ```\n        '
        if client_id and sync_id:
            plex_tv = plextv.PlexTV()
            delete_row = plex_tv.delete_sync(client_id=client_id, sync_id=sync_id)
            if delete_row:
                return {'result': 'success', 'message': 'Synced item deleted successfully.'}
            else:
                return {'result': 'error', 'message': 'Failed to delete synced item.'}
        else:
            return {'result': 'error', 'message': 'Missing client ID and sync ID.'}

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def logs(self, **kwargs):
        if False:
            return 10
        plex_log_files = log_reader.list_plex_logs()
        return serve_template(template_name='logs.html', title='Log', plex_log_files=plex_log_files)

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def get_log(self, logfile='', **kwargs):
        if False:
            print('Hello World!')
        json_data = helpers.process_json_kwargs(json_kwargs=kwargs.get('json_data'))
        log_level = kwargs.get('log_level', '')
        start = json_data['start']
        length = json_data['length']
        order_column = json_data['order'][0]['column']
        order_dir = json_data['order'][0]['dir']
        search_value = json_data['search']['value']
        sortcolumn = 0
        filt = []
        filtered = []
        fa = filt.append
        if logfile == 'tautulli_api':
            filename = logger.FILENAME_API
        elif logfile == 'plex_websocket':
            filename = logger.FILENAME_PLEX_WEBSOCKET
        else:
            filename = logger.FILENAME
        with open(os.path.join(plexpy.CONFIG.LOG_DIR, filename), 'r', encoding='utf-8') as f:
            for l in f.readlines():
                try:
                    temp_loglevel_and_time = l.split(' - ', 1)
                    loglvl = temp_loglevel_and_time[1].split(' ::', 1)[0].strip()
                    msg = helpers.sanitize(l.split(' : ', 1)[1].replace('\n', ''))
                    fa([temp_loglevel_and_time[0], loglvl, msg])
                except IndexError:
                    tl = len(filt) - 1
                    n = len(l) - len(l.lstrip(' '))
                    ll = '&nbsp;' * (2 * n) + helpers.sanitize(l[n:])
                    filt[tl][2] += '<br>' + ll
                    continue
        log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if log_level in log_levels:
            log_levels = log_levels[log_levels.index(log_level):]
            filtered = [row for row in filt if row[1] in log_levels]
        else:
            filtered = filt
        if search_value:
            filtered = [row for row in filtered for column in row if search_value.lower() in column.lower()]
        if order_column == '1':
            sortcolumn = 2
        elif order_column == '2':
            sortcolumn = 1
        filtered.sort(key=lambda x: x[sortcolumn])
        if order_dir == 'desc':
            filtered = filtered[::-1]
        rows = filtered[start:start + length]
        return json.dumps({'recordsFiltered': len(filtered), 'recordsTotal': len(filt), 'data': rows})

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_plex_log(self, logfile='', **kwargs):
        if False:
            i = 10
            return i + 15
        ' Get the PMS logs.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                window (int):           The number of tail lines to return\n                logfile (int):          The name of the Plex log file,\n                                        e.g. "Plex Media Server", "Plex Media Scanner"\n\n            Returns:\n                json:\n                    [["May 08, 2016 09:35:37",\n                      "DEBUG",\n                      "Auth: Came in with a super-token, authorization succeeded."\n                      ],\n                     [...],\n                     [...]\n                     ]\n            ```\n        '
        if kwargs.get('log_type'):
            logfile = 'Plex Media ' + kwargs['log_type'].capitalize()
        window = int(kwargs.get('window', plexpy.CONFIG.PMS_LOGS_LINE_CAP))
        try:
            return {'data': log_reader.get_log_tail(window=window, parsed=True, log_file=logfile)}
        except:
            logger.warn("Unable to retrieve Plex log file '%'." % logfile)
            return []

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @sanitize_out()
    @addtoapi()
    def get_notification_log(self, **kwargs):
        if False:
            while True:
                i = 10
        ' Get the data on the Tautulli notification logs table.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                order_column (str):             "timestamp", "notifier_id", "agent_name", "notify_action",\n                                                "subject_text", "body_text",\n                order_dir (str):                "desc" or "asc"\n                start (int):                    Row to start from, 0\n                length (int):                   Number of items to return, 25\n                search (str):                   A string to search for, "Telegram"\n\n            Returns:\n                json:\n                    {"draw": 1,\n                     "recordsTotal": 1039,\n                     "recordsFiltered": 163,\n                     "data":\n                        [{"agent_id": 13,\n                          "agent_name": "telegram",\n                          "body_text": "DanyKhaleesi69 started playing The Red Woman.",\n                          "id": 1000,\n                          "notify_action": "on_play",\n                          "rating_key": 153037,\n                          "session_key": 147,\n                          "subject_text": "Tautulli (Winterfell-Server)",\n                          "success": 1,\n                          "timestamp": 1462253821,\n                          "user": "DanyKhaleesi69",\n                          "user_id": 8008135\n                          },\n                         {...},\n                         {...}\n                         ]\n                     }\n            ```\n        '
        if not kwargs.get('json_data'):
            dt_columns = [('timestamp', True, True), ('notifier_id', True, True), ('agent_name', True, True), ('notify_action', True, True), ('subject_text', True, True), ('body_text', True, True)]
            kwargs['json_data'] = build_datatables_json(kwargs, dt_columns, 'timestamp')
        data_factory = datafactory.DataFactory()
        notification_logs = data_factory.get_notification_log(kwargs=kwargs)
        return notification_logs

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @sanitize_out()
    @addtoapi()
    def get_newsletter_log(self, **kwargs):
        if False:
            return 10
        ' Get the data on the Tautulli newsletter logs table.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                order_column (str):             "timestamp", "newsletter_id", "agent_name", "notify_action",\n                                                "subject_text", "start_date", "end_date", "uuid"\n                order_dir (str):                "desc" or "asc"\n                start (int):                    Row to start from, 0\n                length (int):                   Number of items to return, 25\n                search (str):                   A string to search for, "Telegram"\n\n            Returns:\n                json:\n                    {"draw": 1,\n                     "recordsTotal": 1039,\n                     "recordsFiltered": 163,\n                     "data":\n                        [{"agent_id": 0,\n                          "agent_name": "recently_added",\n                          "end_date": "2018-03-18",\n                          "id": 7,\n                          "newsletter_id": 1,\n                          "notify_action": "on_cron",\n                          "start_date": "2018-03-05",\n                          "subject_text": "Recently Added to Plex (Winterfell-Server)! (2018-03-18)",\n                          "success": 1,\n                          "timestamp": 1462253821,\n                          "uuid": "7fe4g65i"\n                          },\n                         {...},\n                         {...}\n                         ]\n                     }\n            ```\n        '
        if not kwargs.get('json_data'):
            dt_columns = [('timestamp', True, True), ('newsletter_id', True, True), ('agent_name', True, True), ('notify_action', True, True), ('subject_text', True, True), ('body_text', True, True), ('start_date', True, True), ('end_date', True, True), ('uuid', True, True)]
            kwargs['json_data'] = build_datatables_json(kwargs, dt_columns, 'timestamp')
        data_factory = datafactory.DataFactory()
        newsletter_logs = data_factory.get_newsletter_log(kwargs=kwargs)
        return newsletter_logs

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def delete_notification_log(self, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Delete the Tautulli notification logs.\n\n            ```\n            Required paramters:\n                None\n\n            Optional parameters:\n                None\n\n            Returns:\n                None\n            ```\n        '
        data_factory = datafactory.DataFactory()
        result = data_factory.delete_notification_log()
        res = 'success' if result else 'error'
        msg = 'Cleared notification logs.' if result else 'Failed to clear notification logs.'
        return {'result': res, 'message': msg}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def delete_newsletter_log(self, **kwargs):
        if False:
            print('Hello World!')
        ' Delete the Tautulli newsletter logs.\n\n            ```\n            Required paramters:\n                None\n\n            Optional parameters:\n                None\n\n            Returns:\n                None\n            ```\n        '
        data_factory = datafactory.DataFactory()
        result = data_factory.delete_newsletter_log()
        res = 'success' if result else 'error'
        msg = 'Cleared newsletter logs.' if result else 'Failed to clear newsletter logs.'
        return {'result': res, 'message': msg}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def delete_login_log(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Delete the Tautulli login logs.\n\n            ```\n            Required paramters:\n                None\n\n            Optional parameters:\n                None\n\n            Returns:\n                None\n            ```\n        '
        user_data = users.Users()
        result = user_data.delete_login_log()
        res = 'success' if result else 'error'
        msg = 'Cleared login logs.' if result else 'Failed to clear login logs.'
        return {'result': res, 'message': msg}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def delete_logs(self, logfile='', **kwargs):
        if False:
            while True:
                i = 10
        if logfile == 'tautulli_api':
            filename = logger.FILENAME_API
        elif logfile == 'plex_websocket':
            filename = logger.FILENAME_PLEX_WEBSOCKET
        else:
            filename = logger.FILENAME
        try:
            open(os.path.join(plexpy.CONFIG.LOG_DIR, filename), 'w').close()
            result = 'success'
            msg = 'Cleared the %s file.' % filename
            logger.info(msg)
        except Exception as e:
            result = 'error'
            msg = 'Failed to clear the %s file.' % filename
            logger.exception('Failed to clear the %s file: %s.' % (filename, e))
        return {'result': result, 'message': msg}

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def toggleVerbose(self, **kwargs):
        if False:
            print('Hello World!')
        plexpy.VERBOSE = not plexpy.VERBOSE
        plexpy.CONFIG.VERBOSE_LOGS = plexpy.VERBOSE
        plexpy.CONFIG.write()
        logger.initLogger(console=not plexpy.QUIET, log_dir=plexpy.CONFIG.LOG_DIR, verbose=plexpy.VERBOSE)
        logger.info('Verbose toggled, set to %s', plexpy.VERBOSE)
        logger.debug('If you read this message, debug logging is available')
        raise cherrypy.HTTPRedirect(plexpy.HTTP_ROOT + 'logs')

    @cherrypy.expose
    @requireAuth()
    def log_js_errors(self, page, message, file, line, **kwargs):
        if False:
            while True:
                i = 10
        ' Logs javascript errors from the web interface. '
        logger.error('WebUI :: /%s : %s. (%s:%s)' % (page.rpartition('/')[-1], message, file.rpartition('/')[-1].partition('?')[0], line))
        return 'js error logged.'

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def logFile(self, logfile='', **kwargs):
        if False:
            print('Hello World!')
        if logfile == 'tautulli_api':
            filename = logger.FILENAME_API
        elif logfile == 'plex_websocket':
            filename = logger.FILENAME_PLEX_WEBSOCKET
        else:
            filename = logger.FILENAME
        try:
            with open(os.path.join(plexpy.CONFIG.LOG_DIR, filename), 'r', encoding='utf-8') as f:
                return '<pre>%s</pre>' % f.read()
        except IOError as e:
            return 'Log file not found.'

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def settings(self, **kwargs):
        if False:
            i = 10
            return i + 15
        settings_dict = {}
        for setting in config.SETTINGS:
            settings_dict[setting.lower()] = getattr(plexpy.CONFIG, setting)
        for setting in config.CHECKED_SETTINGS:
            settings_dict[setting.lower()] = checked(getattr(plexpy.CONFIG, setting))
        if plexpy.CONFIG.HTTP_PASSWORD != '':
            settings_dict['http_password'] = '    '
        else:
            settings_dict['http_password'] = ''
        for key in ('home_sections', 'home_stats_cards', 'home_library_cards'):
            settings_dict[key] = json.dumps(settings_dict[key])
        return serve_template(template_name='settings.html', title='Settings', config=settings_dict)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def configUpdate(self, **kwargs):
        if False:
            print('Hello World!')
        first_run = False
        startup_changed = False
        server_changed = False
        reschedule = False
        https_changed = False
        refresh_libraries = False
        refresh_users = False
        if kwargs.pop('first_run', None):
            first_run = True
            server_changed = True
        if not first_run:
            for checked_config in config.CHECKED_SETTINGS:
                checked_config = checked_config.lower()
                if checked_config not in kwargs:
                    kwargs[checked_config] = 0
                else:
                    kwargs[checked_config] = 1
        if kwargs.get('http_password') == '    ':
            del kwargs['http_password']
        else:
            if kwargs.get('http_password', '') != '':
                kwargs['http_password'] = make_hash(kwargs['http_password'])
            kwargs['jwt_update_secret'] = True and (not first_run)
        for (plain_config, use_config) in [(x[4:], x) for x in kwargs if x.startswith('use_')]:
            kwargs[plain_config] = kwargs[use_config]
            del kwargs[use_config]
        if kwargs.get('launch_startup') != plexpy.CONFIG.LAUNCH_STARTUP or kwargs.get('launch_browser') != plexpy.CONFIG.LAUNCH_BROWSER:
            startup_changed = True
        if kwargs.get('check_github') != plexpy.CONFIG.CHECK_GITHUB or kwargs.get('check_github_interval') != str(plexpy.CONFIG.CHECK_GITHUB_INTERVAL) or kwargs.get('refresh_libraries_interval') != str(plexpy.CONFIG.REFRESH_LIBRARIES_INTERVAL) or (kwargs.get('refresh_users_interval') != str(plexpy.CONFIG.REFRESH_USERS_INTERVAL)) or (kwargs.get('pms_update_check_interval') != str(plexpy.CONFIG.PMS_UPDATE_CHECK_INTERVAL)) or (kwargs.get('monitor_pms_updates') != plexpy.CONFIG.MONITOR_PMS_UPDATES) or (kwargs.get('pms_url_manual') != plexpy.CONFIG.PMS_URL_MANUAL) or (kwargs.get('backup_interval') != str(plexpy.CONFIG.BACKUP_INTERVAL)):
            reschedule = True
        if kwargs.get('pms_ssl') != str(plexpy.CONFIG.PMS_SSL) or kwargs.get('pms_is_remote') != str(plexpy.CONFIG.PMS_IS_REMOTE) or kwargs.get('pms_url_manual') != plexpy.CONFIG.PMS_URL_MANUAL:
            server_changed = True
        if kwargs.get('enable_https') and kwargs.get('https_create_cert'):
            if kwargs.get('https_domain') != plexpy.CONFIG.HTTPS_DOMAIN or kwargs.get('https_ip') != plexpy.CONFIG.HTTPS_IP or kwargs.get('https_cert') != plexpy.CONFIG.HTTPS_CERT or (kwargs.get('https_key') != plexpy.CONFIG.HTTPS_KEY):
                https_changed = True
        if kwargs.get('home_sections'):
            for k in list(kwargs.keys()):
                if k.startswith('hsec-'):
                    del kwargs[k]
            kwargs['home_sections'] = kwargs['home_sections'].split(',')
        if kwargs.get('home_stats_cards'):
            for k in list(kwargs.keys()):
                if k.startswith('hscard-'):
                    del kwargs[k]
            kwargs['home_stats_cards'] = kwargs['home_stats_cards'].split(',')
        if kwargs.get('home_library_cards'):
            for k in list(kwargs.keys()):
                if k.startswith('hlcard-'):
                    del kwargs[k]
            kwargs['home_library_cards'] = kwargs['home_library_cards'].split(',')
        if kwargs.pop('server_changed', None) or server_changed:
            server_changed = True
            refresh_users = True
            refresh_libraries = True
        if kwargs.pop('auth_changed', None):
            refresh_users = True
        all_settings = config.SETTINGS + config.CHECKED_SETTINGS
        kwargs = {k: v for (k, v) in kwargs.items() if k.upper() in all_settings}
        if first_run:
            kwargs['first_run_complete'] = 1
        plexpy.CONFIG.process_kwargs(kwargs)
        plexpy.CONFIG.write()
        if startup_changed:
            if common.PLATFORM == 'Windows':
                windows.set_startup()
            elif common.PLATFORM == 'Darwin':
                macos.set_startup()
        if server_changed:
            plextv.get_server_resources()
            if plexpy.WS_CONNECTED:
                web_socket.reconnect()
        if first_run:
            webstart.restart()
            activity_pinger.connect_server(log=True, startup=True)
        if reschedule:
            plexpy.initialize_scheduler()
        if https_changed:
            create_https_certificates(plexpy.CONFIG.HTTPS_CERT, plexpy.CONFIG.HTTPS_KEY)
        if refresh_libraries:
            threading.Thread(target=libraries.refresh_libraries).start()
        if refresh_users:
            threading.Thread(target=users.refresh_users).start()
        return {'result': 'success', 'message': 'Settings saved.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def check_pms_token(self, **kwargs):
        if False:
            i = 10
            return i + 15
        plex_tv = plextv.PlexTV()
        response = plex_tv.get_plextv_resources(return_response=True)
        if not response.ok:
            cherrypy.response.status = 401

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def get_pms_downloads(self, update_channel, **kwargs):
        if False:
            print('Hello World!')
        plex_tv = plextv.PlexTV()
        downloads = plex_tv.get_plex_downloads(update_channel=update_channel)
        return downloads

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def get_server_resources(self, **kwargs):
        if False:
            return 10
        return plextv.get_server_resources(return_server=True, **kwargs)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def backup_config(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Creates a manual backup of the plexpy.db file '
        result = config.make_backup()
        if result:
            return {'result': 'success', 'message': 'Config backup successful.'}
        else:
            return {'result': 'error', 'message': 'Config backup failed.'}

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def get_configuration_table(self, **kwargs):
        if False:
            return 10
        return serve_template(template_name='configuration_table.html')

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def get_scheduler_table(self, **kwargs):
        if False:
            i = 10
            return i + 15
        return serve_template(template_name='scheduler_table.html')

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def get_queue_modal(self, queue=None, **kwargs):
        if False:
            i = 10
            return i + 15
        return serve_template(template_name='queue_modal.html', queue=queue)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def get_server_update_params(self, **kwargs):
        if False:
            return 10
        plex_tv = plextv.PlexTV()
        plexpass = plex_tv.get_plexpass_status()
        update_channel = pmsconnect.PmsConnect().get_server_update_channel()
        return {'plexpass': plexpass, 'pms_platform': common.PMS_PLATFORM_NAME_OVERRIDES.get(plexpy.CONFIG.PMS_PLATFORM, plexpy.CONFIG.PMS_PLATFORM), 'pms_update_channel': plexpy.CONFIG.PMS_UPDATE_CHANNEL, 'pms_update_distro': plexpy.CONFIG.PMS_UPDATE_DISTRO, 'pms_update_distro_build': plexpy.CONFIG.PMS_UPDATE_DISTRO_BUILD}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def backup_db(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Creates a manual backup of the plexpy.db file '
        result = database.make_backup()
        if result:
            return {'result': 'success', 'message': 'Database backup successful.'}
        else:
            return {'result': 'error', 'message': 'Database backup failed.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_notifiers(self, notify_action=None, **kwargs):
        if False:
            print('Hello World!')
        ' Get a list of configured notifiers.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                notify_action (str):        The notification action to filter out\n\n            Returns:\n                json:\n                    [{"id": 1,\n                      "agent_id": 13,\n                      "agent_name": "telegram",\n                      "agent_label": "Telegram",\n                      "friendly_name": "",\n                      "active": 1\n                      }\n                     ]\n            ```\n        '
        result = notifiers.get_notifiers(notify_action=notify_action)
        return result

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def get_notifiers_table(self, **kwargs):
        if False:
            i = 10
            return i + 15
        result = notifiers.get_notifiers()
        return serve_template(template_name='notifiers_table.html', notifiers_list=result)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def delete_notifier(self, notifier_id=None, **kwargs):
        if False:
            while True:
                i = 10
        ' Remove a notifier from the database.\n\n            ```\n            Required parameters:\n                notifier_id (int):        The notifier to delete\n\n            Optional parameters:\n                None\n\n            Returns:\n                None\n            ```\n        '
        result = notifiers.delete_notifier(notifier_id=notifier_id)
        if result:
            return {'result': 'success', 'message': 'Notifier deleted successfully.'}
        else:
            return {'result': 'error', 'message': 'Failed to delete notifier.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_notifier_config(self, notifier_id=None, **kwargs):
        if False:
            print('Hello World!')
        ' Get the configuration for an existing notification agent.\n\n            ```\n            Required parameters:\n                notifier_id (int):        The notifier config to retrieve\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    {"id": 1,\n                     "agent_id": 13,\n                     "agent_name": "telegram",\n                     "agent_label": "Telegram",\n                     "friendly_name": "",\n                     "config": {"incl_poster": 0,\n                                "html_support": 1,\n                                "chat_id": "123456",\n                                "bot_token": "13456789:fio9040NNo04jLEp-4S",\n                                "incl_subject": 1,\n                                "disable_web_preview": 0\n                                },\n                     "config_options": [{...}, ...]\n                     "actions": {"on_play": 0,\n                                 "on_stop": 0,\n                                 ...\n                                 },\n                     "notify_text": {"on_play": {"subject": "...",\n                                                 "body": "..."\n                                                 }\n                                     "on_stop": {"subject": "...",\n                                                 "body": "..."\n                                                 }\n                                     ...\n                                     }\n                     }\n            ```\n        '
        result = notifiers.get_notifier_config(notifier_id=notifier_id, mask_passwords=True)
        return result

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def get_notifier_config_modal(self, notifier_id=None, **kwargs):
        if False:
            return 10
        result = notifiers.get_notifier_config(notifier_id=notifier_id, mask_passwords=True)
        parameters = [{'name': param['name'], 'type': param['type'], 'value': param['value']} for category in common.NOTIFICATION_PARAMETERS for param in category['parameters']]
        return serve_template(template_name='notifier_config.html', notifier=result, parameters=parameters)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def add_notifier_config(self, agent_id=None, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Add a new notification agent.\n\n            ```\n            Required parameters:\n                agent_id (int):           The notification agent to add\n\n            Optional parameters:\n                None\n\n            Returns:\n                None\n            ```\n        '
        result = notifiers.add_notifier_config(agent_id=agent_id, **kwargs)
        if result:
            return {'result': 'success', 'message': 'Added notification agent.', 'notifier_id': result}
        else:
            return {'result': 'error', 'message': 'Failed to add notification agent.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def set_notifier_config(self, notifier_id=None, agent_id=None, **kwargs):
        if False:
            print('Hello World!')
        ' Configure an existing notification agent.\n\n            ```\n            Required parameters:\n                notifier_id (int):        The notifier config to update\n                agent_id (int):           The agent of the notifier\n\n            Optional parameters:\n                Pass all the config options for the agent with the agent prefix:\n                    e.g. For Telegram: telegram_bot_token\n                                       telegram_chat_id\n                                       telegram_disable_web_preview\n                                       telegram_html_support\n                                       telegram_incl_poster\n                                       telegram_incl_subject\n                Notify actions (int):  0 or 1,\n                    e.g. on_play, on_stop, etc.\n                Notify text (str):\n                    e.g. on_play_subject, on_play_body, etc.\n\n            Returns:\n                None\n            ```\n        '
        result = notifiers.set_notifier_config(notifier_id=notifier_id, agent_id=agent_id, **kwargs)
        if result:
            return {'result': 'success', 'message': 'Saved notification agent.'}
        else:
            return {'result': 'error', 'message': 'Failed to save notification agent.'}

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def get_notify_text_preview(self, notify_action='', subject='', body='', agent_id=0, agent_name='', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if str(agent_id).isdigit():
            agent_id = int(agent_id)
        text = []
        media_types = next((a['media_types'] for a in notifiers.available_notification_actions() if a['name'] == notify_action), ())
        for media_type in media_types:
            (test_subject, test_body) = notification_handler.build_notify_text(subject=subject, body=body, notify_action=notify_action, parameters={'media_type': media_type}, agent_id=agent_id, test=True)
            text.append({'media_type': media_type, 'subject': test_subject, 'body': test_body})
        return serve_template(template_name='notifier_text_preview.html', text=text, agent=agent_name)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_notifier_parameters(self, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Get the list of available notification parameters.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    {\n                     }\n            ```\n        '
        parameters = [{'name': param['name'], 'type': param['type'], 'value': param['value']} for category in common.NOTIFICATION_PARAMETERS for param in category['parameters']]
        return parameters

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def send_notification(self, notifier_id=None, subject='Tautulli', body='Test notification', notify_action='', **kwargs):
        if False:
            print('Hello World!')
        ' Send a notification using Tautulli.\n\n            ```\n            Required parameters:\n                notifier_id (int):      The ID number of the notification agent\n                subject (str):          The subject of the message\n                body (str):             The body of the message\n\n            Optional parameters:\n                None\n\n            Returns:\n                None\n            ```\n        '
        cherrypy.response.headers['Cache-Control'] = 'max-age=0,no-cache,no-store'
        test = 'test ' if notify_action == 'test' else ''
        if notifier_id:
            notifier = notifiers.get_notifier_config(notifier_id=notifier_id)
            if notifier:
                logger.debug('Sending %s%s notification.' % (test, notifier['agent_label']))
                notification_handler.add_notifier_each(notifier_id=notifier_id, notify_action=notify_action, subject=subject, body=body, manual_trigger=True, **kwargs)
                return {'result': 'success', 'message': 'Notification queued.'}
            else:
                logger.debug('Unable to send %snotification, invalid notifier_id %s.' % (test, notifier_id))
                return {'result': 'error', 'message': 'Invalid notifier id %s.' % notifier_id}
        else:
            logger.debug('Unable to send %snotification, no notifier_id received.' % test)
            return {'result': 'error', 'message': 'No notifier id received.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def get_browser_notifications(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        result = notifiers.get_browser_notifications()
        if result:
            notifications = result['notifications']
            if notifications:
                return notifications
            else:
                return None
        else:
            logger.warn('Unable to retrieve browser notifications.')
            return None

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def facebook_auth(self, app_id='', app_secret='', redirect_uri='', **kwargs):
        if False:
            i = 10
            return i + 15
        cherrypy.response.headers['Cache-Control'] = 'max-age=0,no-cache,no-store'
        facebook_notifier = notifiers.FACEBOOK()
        url = facebook_notifier._get_authorization(app_id=app_id, app_secret=app_secret, redirect_uri=redirect_uri)
        if url:
            return {'result': 'success', 'msg': 'Confirm Authorization. Check pop-up blocker if no response.', 'url': url}
        else:
            return {'result': 'error', 'msg': 'Failed to retrieve authorization url.'}

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def facebook_redirect(self, code='', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        cherrypy.response.headers['Cache-Control'] = 'max-age=0,no-cache,no-store'
        facebook = notifiers.FACEBOOK()
        access_token = facebook._get_credentials(code)
        if access_token:
            return 'Facebook authorization successful. Tautulli can send notification to Facebook. Your Facebook access token is:<pre>{0}</pre>You may close this page.'.format(access_token)
        else:
            return 'Failed to request authorization from Facebook. Check the Tautulli logs for details.<br />You may close this page.'

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def facebook_retrieve_token(self, **kwargs):
        if False:
            print('Hello World!')
        if plexpy.CONFIG.FACEBOOK_TOKEN == 'temp':
            return {'result': 'waiting'}
        elif plexpy.CONFIG.FACEBOOK_TOKEN:
            token = plexpy.CONFIG.FACEBOOK_TOKEN
            plexpy.CONFIG.FACEBOOK_TOKEN = ''
            return {'result': 'success', 'msg': 'Authorization successful.', 'access_token': token}
        else:
            return {'result': 'error', 'msg': 'Failed to request authorization.'}

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def osxnotifyregister(self, app, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        cherrypy.response.headers['Cache-Control'] = 'max-age=0,no-cache,no-store'
        from osxnotify import registerapp as osxnotify
        (result, msg) = osxnotify.registerapp(app)
        if result:
            osx_notify = notifiers.OSX()
            osx_notify.notify(subject='Registered', body='Success :-)', subtitle=result)
        else:
            logger.warn(msg)
        return msg

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def zapier_test_hook(self, zapier_hook='', **kwargs):
        if False:
            i = 10
            return i + 15
        success = notifiers.ZAPIER(config={'hook': zapier_hook})._test_hook()
        if success:
            return {'result': 'success', 'msg': 'Test Zapier webhook sent.'}
        else:
            return {'result': 'error', 'msg': 'Failed to send test Zapier webhook.'}

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def set_notification_config(self, **kwargs):
        if False:
            i = 10
            return i + 15
        for (plain_config, use_config) in [(x[4:], x) for x in kwargs if x.startswith('use_')]:
            kwargs[plain_config] = kwargs[use_config]
            del kwargs[use_config]
        plexpy.CONFIG.process_kwargs(kwargs)
        plexpy.CONFIG.write()
        cherrypy.response.status = 200

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def get_mobile_devices_table(self, **kwargs):
        if False:
            i = 10
            return i + 15
        result = mobile_app.get_mobile_devices()
        return serve_template(template_name='mobile_devices_table.html', devices_list=result)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def verify_mobile_device(self, device_token='', cancel=False, **kwargs):
        if False:
            print('Hello World!')
        if helpers.bool_true(cancel):
            mobile_app.set_temp_device_token(device_token, remove=True)
            return {'result': 'error', 'message': 'Device registration cancelled.'}
        result = mobile_app.get_temp_device_token(device_token)
        if result is True:
            mobile_app.set_temp_device_token(device_token, remove=True)
            return {'result': 'success', 'message': 'Device registered successfully.', 'data': result}
        else:
            return {'result': 'error', 'message': 'Device not registered.'}

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def get_mobile_device_config_modal(self, mobile_device_id=None, **kwargs):
        if False:
            i = 10
            return i + 15
        result = mobile_app.get_mobile_device_config(mobile_device_id=mobile_device_id)
        return serve_template(template_name='mobile_device_config.html', device=result)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def set_mobile_device_config(self, mobile_device_id=None, **kwargs):
        if False:
            return 10
        ' Configure an existing notification agent.\n\n            ```\n            Required parameters:\n                mobile_device_id (int):        The mobile device config to update\n\n            Optional parameters:\n                friendly_name (str):           A friendly name to identify the mobile device\n\n            Returns:\n                None\n            ```\n        '
        result = mobile_app.set_mobile_device_config(mobile_device_id=mobile_device_id, **kwargs)
        if result:
            return {'result': 'success', 'message': 'Saved mobile device.'}
        else:
            return {'result': 'error', 'message': 'Failed to save mobile device.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def delete_mobile_device(self, mobile_device_id=None, device_id=None, **kwargs):
        if False:
            return 10
        ' Remove a mobile device from the database.\n\n            ```\n            Required parameters:\n                mobile_device_id (int):        The mobile device database id to delete, OR\n                device_id (str):               The unique device identifier for the mobile device\n\n            Optional parameters:\n                None\n\n            Returns:\n                None\n            ```\n        '
        result = mobile_app.delete_mobile_device(mobile_device_id=mobile_device_id, device_id=device_id)
        if result:
            return {'result': 'success', 'message': 'Deleted mobile device.'}
        else:
            return {'result': 'error', 'message': 'Failed to delete device.'}

    @cherrypy.config(**{'response.timeout': 3600})
    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def import_database(self, app=None, database_file=None, database_path=None, method=None, backup=False, table_name=None, import_ignore_interval=0, **kwargs):
        if False:
            return 10
        ' Import a Tautulli, PlexWatch, or Plexivity database into Tautulli.\n\n            ```\n            Required parameters:\n                app (str):                      "tautulli" or "plexwatch" or "plexivity"\n                database_file (file):           The database file to import (multipart/form-data)\n                or\n                database_path (str):            The full path to the database file to import\n                method (str):                   For Tautulli only, "merge" or "overwrite"\n                table_name (str):               For PlexWatch or Plexivity only, "processed" or "grouped"\n\n\n            Optional parameters:\n                backup (bool):                  For Tautulli only, true or false whether to backup\n                                                the current database before importing\n                import_ignore_interval (int):   For PlexWatch or Plexivity only, the minimum number\n                                                of seconds for a stream to import\n\n            Returns:\n                json:\n                    {"result": "success",\n                     "message": "Database import has started. Check the logs to monitor any problems."\n                     }\n            ```\n        '
        if not app:
            return {'result': 'error', 'message': 'No app specified for import'}
        if database_path:
            database_file_name = os.path.basename(database_path)
            database_cache_path = os.path.join(plexpy.CONFIG.CACHE_DIR, database_file_name + '.import.db')
            logger.info("Received database file '%s' for import. Saving to cache: %s", database_file_name, database_cache_path)
            database_path = shutil.copyfile(database_path, database_cache_path)
        elif database_file:
            database_path = os.path.join(plexpy.CONFIG.CACHE_DIR, database_file.filename + '.import.db')
            logger.info("Received database file '%s' for import. Saving to cache: %s", database_file.filename, database_path)
            with open(database_path, 'wb') as f:
                while True:
                    data = database_file.file.read(8192)
                    if not data:
                        break
                    f.write(data)
        if not database_path:
            return {'result': 'error', 'message': 'No database specified for import'}
        if app.lower() == 'tautulli':
            db_check_msg = database.validate_database(database=database_path)
            if db_check_msg == 'success':
                threading.Thread(target=database.import_tautulli_db, kwargs={'database': database_path, 'method': method, 'backup': helpers.bool_true(backup)}).start()
                return {'result': 'success', 'message': 'Database import has started. Check the logs to monitor any problems.'}
            else:
                if database_file:
                    helpers.delete_file(database_path)
                return {'result': 'error', 'message': db_check_msg}
        elif app.lower() == 'plexwatch':
            db_check_msg = plexwatch_import.validate_database(database_file=database_path, table_name=table_name)
            if db_check_msg == 'success':
                threading.Thread(target=plexwatch_import.import_from_plexwatch, kwargs={'database_file': database_path, 'table_name': table_name, 'import_ignore_interval': import_ignore_interval}).start()
                return {'result': 'success', 'message': 'Database import has started. Check the logs to monitor any problems.'}
            else:
                if database_file:
                    helpers.delete_file(database_path)
                return {'result': 'error', 'message': db_check_msg}
        elif app.lower() == 'plexivity':
            db_check_msg = plexivity_import.validate_database(database_file=database_path, table_name=table_name)
            if db_check_msg == 'success':
                threading.Thread(target=plexivity_import.import_from_plexivity, kwargs={'database_file': database_path, 'table_name': table_name, 'import_ignore_interval': import_ignore_interval}).start()
                return {'result': 'success', 'message': 'Database import has started. Check the logs to monitor any problems.'}
            else:
                if database_file:
                    helpers.delete_file(database_path)
                return {'result': 'error', 'message': db_check_msg}
        else:
            return {'result': 'error', 'message': 'App not recognized for import'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def import_config(self, config_file=None, config_path=None, backup=False, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Import a Tautulli config file.\n\n            ```\n            Required parameters:\n                config_file (file):             The config file to import (multipart/form-data)\n                or\n                config_path (str):              The full path to the config file to import\n\n\n            Optional parameters:\n                backup (bool):                  true or false whether to backup\n                                                the current config before importing\n\n            Returns:\n                json:\n                    {"result": "success",\n                     "message": "Config import has started. Check the logs to monitor any problems. "\n                                "Tautulli will restart automatically."\n                     }\n            ```\n        '
        if database.IS_IMPORTING:
            return {'result': 'error', 'message': 'Database import is in progress. Please wait until it is finished to import a config.'}
        if config_file:
            config_path = os.path.join(plexpy.CONFIG.CACHE_DIR, config_file.filename + '.import.ini')
            logger.info("Received config file '%s' for import. Saving to cache '%s'.", config_file.filename, config_path)
            with open(config_path, 'wb') as f:
                while True:
                    data = config_file.file.read(8192)
                    if not data:
                        break
                    f.write(data)
        if not config_path:
            return {'result': 'error', 'message': 'No config specified for import'}
        config.set_import_thread(config=config_path, backup=helpers.bool_true(backup))
        return {'result': 'success', 'message': 'Config import has started. Check the logs to monitor any problems. Tautulli will restart automatically.'}

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def import_database_tool(self, app=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if app == 'tautulli':
            return serve_template(template_name='app_import.html', title='Import Tautulli Database', app='Tautulli')
        elif app == 'plexwatch':
            return serve_template(template_name='app_import.html', title='Import PlexWatch Database', app='PlexWatch')
        elif app == 'plexivity':
            return serve_template(template_name='app_import.html', title='Import Plexivity Database', app='Plexivity')
        logger.warn('No app specified for import.')
        return

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def import_config_tool(self, **kwargs):
        if False:
            print('Hello World!')
        return serve_template(template_name='config_import.html', title='Import Tautulli Configuration')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def browse_path(self, key=None, path=None, filter_ext=''):
        if False:
            print('Hello World!')
        if key:
            path = base64.b64decode(key).decode('UTF-8')
        if not path:
            path = plexpy.DATA_DIR
        data = helpers.browse_path(path=path, filter_ext=filter_ext)
        if data:
            return {'result': 'success', 'path': path, 'data': data}
        else:
            return {'result': 'error', 'message': 'Invalid path.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_server_id(self, hostname=None, port=None, identifier=None, ssl=0, remote=0, manual=0, get_url=False, test_websocket=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        " Get the PMS server identifier.\n\n            ```\n            Required parameters:\n                hostname (str):     'localhost' or '192.160.0.10'\n                port (int):         32400\n\n            Optional parameters:\n                ssl (int):          0 or 1\n                remote (int):       0 or 1\n\n            Returns:\n                json:\n                    {'identifier': '08u2phnlkdshf890bhdlksghnljsahgleikjfg9t'}\n            ```\n        "
        ssl = helpers.bool_true(ssl)
        if not identifier and hostname and port:
            plex_tv = plextv.PlexTV()
            servers = plex_tv.discover()
            ip_address = get_ip(hostname)
            for server in servers:
                if (server['ip'] == hostname or server['ip'] == ip_address) and server['port'] == port:
                    identifier = server['clientIdentifier']
                    break
            if not identifier:
                scheme = 'https' if ssl else 'http'
                url = '{scheme}://{hostname}:{port}'.format(scheme=scheme, hostname=hostname, port=port)
                uri = '/identity'
                request_handler = http_handler.HTTPHandler(urls=url, ssl_verify=False)
                request = request_handler.make_request(uri=uri, request_type='GET', output_format='xml')
                if request:
                    xml_head = request.getElementsByTagName('MediaContainer')[0]
                    identifier = xml_head.getAttribute('machineIdentifier')
        result = {'identifier': identifier}
        if identifier:
            if helpers.bool_true(get_url):
                server = self.get_server_resources(pms_ip=hostname, pms_port=port, pms_ssl=ssl, pms_is_remote=remote, pms_url_manual=manual, pms_identifier=identifier)
                result['url'] = server['pms_url']
                result['ws'] = None
                if helpers.bool_true(test_websocket):
                    ws_url = result['url'].replace('http', 'ws', 1) + '/:/websockets/notifications'
                    header = ['X-Plex-Token: %s' % plexpy.CONFIG.PMS_TOKEN]
                    if ssl:
                        secure = 'secure '
                        if plexpy.CONFIG.VERIFY_SSL_CERT:
                            sslopt = {'ca_certs': certifi.where()}
                        else:
                            sslopt = {'cert_reqs': _ssl.CERT_NONE}
                    else:
                        secure = ''
                        sslopt = None
                    logger.debug('Testing %swebsocket connection...' % secure)
                    try:
                        test_ws = websocket.create_connection(ws_url, header=header, sslopt=sslopt)
                        test_ws.close()
                        logger.debug('Websocket connection test successful.')
                        result['ws'] = True
                    except (websocket.WebSocketException, IOError, Exception) as e:
                        logger.error('Websocket connection test failed: %s' % e)
                        result['ws'] = False
            return result
        else:
            logger.warn('Unable to retrieve the PMS identifier.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_server_info(self, **kwargs):
        if False:
            while True:
                i = 10
        ' Get the PMS server information.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    {"pms_identifier": "08u2phnlkdshf890bhdlksghnljsahgleikjfg9t",\n                     "pms_ip": "10.10.10.1",\n                     "pms_is_remote": 0,\n                     "pms_name": "Winterfell-Server",\n                     "pms_platform": "Windows",\n                     "pms_plexpass": 1,\n                     "pms_port": 32400,\n                     "pms_ssl": 0,\n                     "pms_url": "http://10.10.10.1:32400",\n                     "pms_url_manual": 0,\n                     "pms_version": "1.20.0.3133-fede5bdc7"\n                    }\n            ```\n        '
        server = plextv.get_server_resources(return_info=True)
        server.pop('pms_is_cloud', None)
        return server

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_server_pref(self, pref=None, **kwargs):
        if False:
            return 10
        ' Get a specified PMS server preference.\n\n            ```\n            Required parameters:\n                pref (str):         Name of preference\n\n            Returns:\n                string:             Value of preference\n            ```\n        '
        pms_connect = pmsconnect.PmsConnect()
        result = pms_connect.get_server_pref(pref=pref)
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_server_pref.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def generate_api_key(self, device=None, **kwargs):
        if False:
            while True:
                i = 10
        apikey = ''
        while not apikey or apikey == plexpy.CONFIG.API_KEY or mobile_app.get_mobile_device_by_token(device_token=apikey):
            if sys.version_info >= (3, 6):
                apikey = secrets.token_urlsafe(24)
            else:
                apikey = plexpy.generate_uuid()
        logger.info('New API key generated.')
        logger._BLACKLIST_WORDS.add(apikey)
        if helpers.bool_true(device):
            mobile_app.set_temp_device_token(apikey, add=True)
        return apikey

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def update_check(self, **kwargs):
        if False:
            while True:
                i = 10
        ' Check for Tautulli updates.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                None\n\n            Returns:\n                json\n                    {"result": "success",\n                     "update": true,\n                     "message": "An update for Tautulli is available."\n                    }\n            ```\n        '
        versioncheck.check_update()
        if plexpy.UPDATE_AVAILABLE is None:
            update = {'result': 'error', 'update': None, 'message': 'You are running an unknown version of Tautulli.'}
        elif plexpy.UPDATE_AVAILABLE == 'release':
            update = {'result': 'success', 'update': True, 'release': True, 'message': 'A new release (%s) of Tautulli is available.' % plexpy.LATEST_RELEASE, 'current_release': plexpy.common.RELEASE, 'latest_release': plexpy.LATEST_RELEASE, 'release_url': helpers.anon_url('https://github.com/%s/%s/releases/tag/%s' % (plexpy.CONFIG.GIT_USER, plexpy.CONFIG.GIT_REPO, plexpy.LATEST_RELEASE))}
        elif plexpy.UPDATE_AVAILABLE == 'commit':
            update = {'result': 'success', 'update': True, 'release': False, 'message': 'A newer version of Tautulli is available.', 'current_version': plexpy.CURRENT_VERSION, 'latest_version': plexpy.LATEST_VERSION, 'commits_behind': plexpy.COMMITS_BEHIND, 'compare_url': helpers.anon_url('https://github.com/%s/%s/compare/%s...%s' % (plexpy.CONFIG.GIT_USER, plexpy.CONFIG.GIT_REPO, plexpy.CURRENT_VERSION, plexpy.LATEST_VERSION))}
        else:
            update = {'result': 'success', 'update': False, 'message': 'Tautulli is up to date.'}
        if plexpy.DOCKER or plexpy.SNAP or plexpy.FROZEN:
            update['install_type'] = plexpy.INSTALL_TYPE
        return update

    def do_state_change(self, signal, title, timer, **kwargs):
        if False:
            i = 10
            return i + 15
        message = title
        quote = self.random_arnold_quotes()
        if signal:
            plexpy.SIGNAL = signal
        if plexpy.CONFIG.HTTP_ROOT.strip('/'):
            new_http_root = '/' + plexpy.CONFIG.HTTP_ROOT.strip('/') + '/'
        else:
            new_http_root = '/'
        return serve_template(template_name='shutdown.html', signal=signal, title=title, new_http_root=new_http_root, message=message, timer=timer, quote=quote)

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def shutdown(self, **kwargs):
        if False:
            print('Hello World!')
        return self.do_state_change('shutdown', 'Shutting Down', 15)

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def restart(self, **kwargs):
        if False:
            while True:
                i = 10
        return self.do_state_change('restart', 'Restarting', 30)

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def update(self, **kwargs):
        if False:
            while True:
                i = 10
        if plexpy.PYTHON2:
            raise cherrypy.HTTPRedirect(plexpy.HTTP_ROOT + 'home?update=python2')
        if plexpy.DOCKER or plexpy.SNAP:
            raise cherrypy.HTTPRedirect(plexpy.HTTP_ROOT + 'home')
        plexpy.CONFIG.UPDATE_SHOW_CHANGELOG = 1
        plexpy.CONFIG.write()
        return self.do_state_change('update', 'Updating', 120)

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def checkout_git_branch(self, git_remote=None, git_branch=None, **kwargs):
        if False:
            print('Hello World!')
        if git_branch == plexpy.CONFIG.GIT_BRANCH:
            logger.error('Already on the %s branch' % git_branch)
            raise cherrypy.HTTPRedirect(plexpy.HTTP_ROOT + 'home')
        plexpy.CONFIG.GIT_REMOTE = git_remote
        plexpy.CONFIG.GIT_BRANCH = git_branch
        plexpy.CONFIG.write()
        return self.do_state_change('checkout', 'Switching Git Branches', 120)

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def reset_git_install(self, **kwargs):
        if False:
            i = 10
            return i + 15
        return self.do_state_change('reset', 'Resetting to {}'.format(common.RELEASE), 120)

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def restart_import_config(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if config.IMPORT_THREAD:
            config.IMPORT_THREAD.start()
        return self.do_state_change(None, 'Importing a Config', 15)

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def get_changelog(self, latest_only=False, since_prev_release=False, update_shown=False, **kwargs):
        if False:
            print('Hello World!')
        latest_only = helpers.bool_true(latest_only)
        since_prev_release = helpers.bool_true(since_prev_release)
        if since_prev_release and plexpy.PREV_RELEASE == common.RELEASE:
            latest_only = True
            since_prev_release = False
        if helpers.bool_true(update_shown):
            plexpy.CONFIG.UPDATE_SHOW_CHANGELOG = 0
            plexpy.CONFIG.write()
        return versioncheck.read_changelog(latest_only=latest_only, since_prev_release=since_prev_release)

    @cherrypy.expose
    @requireAuth()
    def info(self, rating_key=None, guid=None, source=None, section_id=None, user_id=None, **kwargs):
        if False:
            while True:
                i = 10
        if rating_key and (not str(rating_key).isdigit()):
            raise cherrypy.HTTPRedirect(plexpy.HTTP_ROOT)
        metadata = None
        config = {'pms_identifier': plexpy.CONFIG.PMS_IDENTIFIER, 'pms_web_url': plexpy.CONFIG.PMS_WEB_URL}
        if user_id:
            user_data = users.Users()
            user_info = user_data.get_details(user_id=user_id)
        else:
            user_info = {}
        if rating_key:
            pms_connect = pmsconnect.PmsConnect()
            metadata = pms_connect.get_metadata_details(rating_key=rating_key, section_id=section_id)
        if not metadata and source == 'history':
            data_factory = datafactory.DataFactory()
            metadata = data_factory.get_metadata_details(rating_key=rating_key, guid=guid)
        if metadata:
            data_factory = datafactory.DataFactory()
            poster_info = data_factory.get_poster_info(metadata=metadata)
            metadata.update(poster_info)
            lookup_info = data_factory.get_lookup_info(metadata=metadata)
            metadata.update(lookup_info)
        if metadata:
            if metadata['section_id'] and (not allow_session_library(metadata['section_id'])):
                raise cherrypy.HTTPRedirect(plexpy.HTTP_ROOT)
            return serve_template(template_name='info.html', metadata=metadata, title='Info', config=config, source=source, user_info=user_info)
        elif get_session_user_id():
            raise cherrypy.HTTPRedirect(plexpy.HTTP_ROOT)
        else:
            return self.update_metadata(rating_key)

    @cherrypy.expose
    @requireAuth()
    def get_item_children(self, rating_key='', media_type=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pms_connect = pmsconnect.PmsConnect()
        result = pms_connect.get_item_children(rating_key=rating_key, media_type=media_type)
        if result:
            return serve_template(template_name='info_children_list.html', data=result, media_type=media_type, title='Children List')
        else:
            logger.warn('Unable to retrieve data for get_item_children.')
            return serve_template(template_name='info_children_list.html', data=None, title='Children List')

    @cherrypy.expose
    @requireAuth()
    def get_item_children_related(self, rating_key='', title='', **kwargs):
        if False:
            while True:
                i = 10
        pms_connect = pmsconnect.PmsConnect()
        result = pms_connect.get_item_children_related(rating_key=rating_key)
        if result:
            return serve_template(template_name='info_collection_list.html', data=result, title=title)
        else:
            return serve_template(template_name='info_collection_list.html', data=None, title=title)

    @cherrypy.expose
    @requireAuth()
    def item_watch_time_stats(self, rating_key=None, media_type=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if rating_key:
            item_data = datafactory.DataFactory()
            result = item_data.get_watch_time_stats(rating_key=rating_key, media_type=media_type)
        else:
            result = None
        if result:
            return serve_template(template_name='user_watch_time_stats.html', data=result, title='Watch Stats')
        else:
            logger.warn('Unable to retrieve data for item_watch_time_stats.')
            return serve_template(template_name='user_watch_time_stats.html', data=None, title='Watch Stats')

    @cherrypy.expose
    @requireAuth()
    def item_user_stats(self, rating_key=None, media_type=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if rating_key:
            item_data = datafactory.DataFactory()
            result = item_data.get_user_stats(rating_key=rating_key, media_type=media_type)
        else:
            result = None
        if result:
            return serve_template(template_name='library_user_stats.html', data=result, title='Player Stats')
        else:
            logger.warn('Unable to retrieve data for item_user_stats.')
            return serve_template(template_name='library_user_stats.html', data=None, title='Player Stats')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_item_watch_time_stats(self, rating_key=None, media_type=None, grouping=None, query_days=None, **kwargs):
        if False:
            return 10
        '  Get the watch time stats for the media item.\n\n            ```\n            Required parameters:\n                rating_key (str):       Rating key of the item\n\n            Optional parameters:\n                media_type (str):       Media type of the item (only required for a collection)\n                grouping (int):         0 or 1\n                query_days (str):       Comma separated days, e.g. "1,7,30,0"\n\n            Returns:\n                json:\n                    [\n                        {\n                            "query_days": 1,\n                            "total_time": 0,\n                            "total_plays": 0\n                        },\n                        {\n                            "query_days": 7,\n                            "total_time": 0,\n                            "total_plays": 0\n                        },\n                        {\n                            "query_days": 30,\n                            "total_time": 0,\n                            "total_plays": 0\n                        },\n                        {\n                            "query_days": 0,\n                            "total_time": 57776,\n                            "total_plays": 13\n                        }\n                    ]\n            ```\n        '
        grouping = helpers.bool_true(grouping, return_none=True)
        if rating_key:
            item_data = datafactory.DataFactory()
            result = item_data.get_watch_time_stats(rating_key=rating_key, media_type=media_type, grouping=grouping, query_days=query_days)
            if result:
                return result
            else:
                logger.warn('Unable to retrieve data for get_item_watch_time_stats.')
                return result
        else:
            logger.warn('Item watch time stats requested but no rating_key received.')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_item_user_stats(self, rating_key=None, media_type=None, grouping=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '  Get the user stats for the media item.\n\n            ```\n            Required parameters:\n                rating_key (str):       Rating key of the item\n\n            Optional parameters:\n                media_type (str):       Media type of the item (only required for a collection)\n                grouping (int):         0 or 1\n\n            Returns:\n                json:\n                    [\n                        {\n                            "friendly_name": "Jon Snow",\n                            "user_id": 1601089,\n                            "user_thumb": "",\n                            "username": "jsnow@thewinteriscoming.com",\n                            "total_plays": 6,\n                            "total_time": 28743\n                        },\n                        {\n                            "friendly_name": "DanyKhaleesi69",\n                            "user_id": 8008135,\n                            "user_thumb": "",\n                            "username": "DanyKhaleesi69",\n                            "total_plays": 5,\n                            "total_time": 18583\n                        }\n                    ]\n            ```\n        '
        grouping = helpers.bool_true(grouping, return_none=True)
        if rating_key:
            item_data = datafactory.DataFactory()
            result = item_data.get_user_stats(rating_key=rating_key, media_type=media_type, grouping=grouping)
            if result:
                return result
            else:
                logger.warn('Unable to retrieve data for get_item_user_stats.')
                return result
        else:
            logger.warn('Item user stats requested but no rating_key received.')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi('get_children_metadata')
    def get_children_metadata_details(self, rating_key='', media_type=None, **kwargs):
        if False:
            while True:
                i = 10
        ' Get the metadata for the children of a media item.\n\n            ```\n            Required parameters:\n                rating_key (str):       Rating key of the item\n                media_type (str):       Media type of the item\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    {"children_count": 9,\n                     "children_type": "season",\n                     "title": "Game of Thrones",\n                     "children_list": [\n                         {...},\n                         {"actors": [],\n                          "added_at": "1403553078",\n                          "art": "/library/metadata/1219/art/1562110346",\n                          "audience_rating": "",\n                          "audience_rating_image": "",\n                          "banner": "",\n                          "collections": [],\n                          "content_rating": "",\n                          "directors": [],\n                          "duration": "",\n                          "full_title": "Season 1"\n                          "genres": [],\n                          "grandparent_rating_key": "",\n                          "grandparent_thumb": "",\n                          "grandparent_title": "",\n                          "guid": "com.plexapp.agents.thetvdb://121361/1?lang=en",\n                          "labels": [],\n                          "last_viewed_at": "1589992348",\n                          "library_name": "TV Shows",\n                          "media_index": "1",\n                          "media_type": "season",\n                          "original_title": "",\n                          "originally_available_at": "",\n                          "parent_media_index": "1",\n                          "parent_rating_key": "1219",\n                          "parent_thumb": "/library/metadata/1219/thumb/1562110346",\n                          "parent_title": "Game of Thrones",\n                          "rating": "",\n                          "rating_image": "",\n                          "rating_key": "1220",\n                          "section_id": "2",\n                          "sort_title": "",\n                          "studio": "",\n                          "summary": "",\n                          "tagline": "",\n                          "thumb": "/library/metadata/1220/thumb/1602176313",\n                          "title": "Season 1",\n                          "updated_at": "1602176313",\n                          "user_rating": "",\n                          "writers": [],\n                          "year": ""\n                          },\n                          {...},\n                          {...}\n                         ]\n                     }\n            ```\n        '
        pms_connect = pmsconnect.PmsConnect()
        metadata = pms_connect.get_item_children(rating_key=rating_key, media_type=media_type)
        if metadata:
            return metadata
        else:
            logger.warn('Unable to retrieve data for get_children_metadata_details.')
            return metadata

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi('notify_recently_added')
    def send_manual_on_created(self, notifier_id='', rating_key='', **kwargs):
        if False:
            i = 10
            return i + 15
        ' Send a recently added notification using Tautulli.\n\n            ```\n            Required parameters:\n                rating_key (int):       The rating key for the media\n\n            Optional parameters:\n                notifier_id (int):      The ID number of the notification agent.\n                                        The notification will send to all enabled notification agents if notifier id is not provided.\n\n            Returns:\n                json\n                    {"result": "success",\n                     "message": "Notification queued."\n                    }\n            ```\n        '
        if rating_key:
            pms_connect = pmsconnect.PmsConnect()
            metadata = pms_connect.get_metadata_details(rating_key=rating_key)
            data = {'timeline_data': metadata, 'notify_action': 'on_created', 'manual_trigger': True}
            if metadata['media_type'] not in ('movie', 'episode', 'track'):
                children = pms_connect.get_item_children(rating_key=rating_key)
                child_keys = [child['rating_key'] for child in children['children_list'] if child['rating_key']]
                data['child_keys'] = child_keys
            if notifier_id:
                data['notifier_id'] = notifier_id
            plexpy.NOTIFY_QUEUE.put(data)
            return {'result': 'success', 'message': 'Notification queued.'}
        else:
            return {'result': 'error', 'message': 'Notification failed.'}

    @cherrypy.expose
    def pms_image_proxy(self, **kwargs):
        if False:
            return 10
        ' See real_pms_image_proxy docs string'
        refresh = False
        if kwargs.get('refresh') or 'no-cache' in cherrypy.request.headers.get('Cache-Control', ''):
            refresh = False if get_session_user_id() else True
        kwargs['refresh'] = refresh
        return self.real_pms_image_proxy(**kwargs)

    @addtoapi('pms_image_proxy')
    def real_pms_image_proxy(self, img=None, rating_key=None, width=750, height=1000, opacity=100, background='000000', blur=0, img_format='png', fallback=None, refresh=False, clip=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Gets an image from the PMS and saves it to the image cache directory.\n\n            ```\n            Required parameters:\n                img (str):              /library/metadata/153037/thumb/1462175060\n                or\n                rating_key (str):       54321\n\n            Optional parameters:\n                width (str):            300\n                height (str):           450\n                opacity (str):          25\n                background (str):       Hex color, e.g. 282828\n                blur (str):             3\n                img_format (str):       png\n                fallback (str):         "poster", "cover", "art", "poster-live", "art-live", "art-live-full", "user"\n                refresh (bool):         True or False whether to refresh the image cache\n                return_hash (bool):     True or False to return the self-hosted image hash instead of the image\n\n            Returns:\n                None\n            ```\n        '
        cherrypy.response.headers['Cache-Control'] = 'max-age=2592000'
        if isinstance(img, str) and img.startswith('interfaces/default/images'):
            fp = os.path.join(plexpy.PROG_DIR, 'data', img)
            return serve_file(path=fp, content_type='image/png')
        if not img and (not rating_key):
            if fallback in common.DEFAULT_IMAGES:
                fbi = common.DEFAULT_IMAGES[fallback]
                fp = os.path.join(plexpy.PROG_DIR, 'data', fbi)
                return serve_file(path=fp, content_type='image/png')
            logger.warn('No image input received.')
            return
        return_hash = helpers.bool_true(kwargs.get('return_hash'))
        if rating_key and (not img):
            if fallback and fallback.startswith('art'):
                img = '/library/metadata/{}/art'.format(rating_key)
            else:
                img = '/library/metadata/{}/thumb'.format(rating_key)
        if img and (not img.startswith('http')):
            parts = 5
            if img.startswith('/playlists'):
                parts -= 1
            rating_key_idx = parts - 2
            parts += int('composite' in img)
            img_split = img.split('/')
            img = '/'.join(img_split[:parts])
            img_rating_key = img_split[rating_key_idx]
            if rating_key != img_rating_key:
                rating_key = img_rating_key
        img_hash = notification_handler.set_hash_image_info(img=img, rating_key=rating_key, width=width, height=height, opacity=opacity, background=background, blur=blur, fallback=fallback, add_to_db=return_hash)
        if return_hash:
            return {'img_hash': img_hash}
        fp = '{}.{}'.format(img_hash, img_format)
        c_dir = os.path.join(plexpy.CONFIG.CACHE_DIR, 'images')
        ffp = os.path.join(c_dir, fp)
        if not os.path.exists(c_dir):
            os.mkdir(c_dir)
        clip = helpers.bool_true(clip)
        try:
            if not plexpy.CONFIG.CACHE_IMAGES or refresh or 'indexes' in img:
                raise NotFound
            return serve_file(path=ffp, content_type='image/png')
        except NotFound:
            try:
                pms_connect = pmsconnect.PmsConnect()
                pms_connect.request_handler._silent = True
                result = pms_connect.get_image(img=img, width=width, height=height, opacity=opacity, background=background, blur=blur, img_format=img_format, clip=clip, refresh=refresh)
                if result and result[0]:
                    cherrypy.response.headers['Content-type'] = result[1]
                    if plexpy.CONFIG.CACHE_IMAGES and 'indexes' not in img:
                        with open(ffp, 'wb') as f:
                            f.write(result[0])
                    return result[0]
                else:
                    raise Exception('PMS image request failed')
            except Exception as e:
                logger.warn('Failed to get image %s, falling back to %s.' % (img, fallback))
                cherrypy.response.headers['Cache-Control'] = 'max-age=0,no-cache,no-store'
                if fallback in common.DEFAULT_IMAGES:
                    fbi = common.DEFAULT_IMAGES[fallback]
                    fp = os.path.join(plexpy.PROG_DIR, 'data', fbi)
                    return serve_file(path=fp, content_type='image/png')
                elif fallback:
                    return self.real_pms_image_proxy(img=fallback, rating_key=None, width=width, height=height, opacity=opacity, background=background, blur=blur, img_format=img_format, fallback=None, refresh=refresh, clip=clip, **kwargs)

    @cherrypy.expose
    def image(self, *args, **kwargs):
        if False:
            return 10
        if args:
            cherrypy.response.headers['Cache-Control'] = 'max-age=3600'
            if len(args) >= 2 and args[0] == 'images':
                resource_dir = os.path.join(str(plexpy.PROG_DIR), 'data/interfaces/default/')
                try:
                    return serve_file(path=os.path.join(resource_dir, *args), content_type='image/png')
                except NotFound:
                    return
            img_hash = args[0].split('.')[0]
            if img_hash in common.DEFAULT_IMAGES:
                fbi = common.DEFAULT_IMAGES[img_hash]
                fp = os.path.join(plexpy.PROG_DIR, 'data', fbi)
                return serve_file(path=fp, content_type='image/png')
            img_info = notification_handler.get_hash_image_info(img_hash=img_hash)
            if img_info:
                kwargs.update(img_info)
                return self.real_pms_image_proxy(refresh=True, **kwargs)
        return

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    @addtoapi()
    def download_config(self, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Download the Tautulli configuration file. '
        config_file = config.FILENAME
        config_copy = os.path.join(plexpy.CONFIG.CACHE_DIR, config_file)
        try:
            plexpy.CONFIG.write()
            shutil.copyfile(plexpy.CONFIG_FILE, config_copy)
        except:
            pass
        try:
            cfg = config.Config(config_copy)
            for key in config._DO_NOT_DOWNLOAD_KEYS:
                setattr(cfg, key, '')
            cfg.write()
        except:
            cherrypy.response.status = 500
            return 'Error downloading config. Check the logs.'
        return serve_download(config_copy, name=config_file)

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    @addtoapi()
    def download_database(self, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Download the Tautulli database file. '
        database_file = database.FILENAME
        database_copy = os.path.join(plexpy.CONFIG.CACHE_DIR, database_file)
        try:
            db = database.MonitorDatabase()
            db.connection.execute('begin immediate')
            shutil.copyfile(plexpy.DB_FILE, database_copy)
            db.connection.rollback()
        except:
            pass
        db = database.MonitorDatabase(database_copy)
        try:
            db.action('UPDATE users SET user_token = NULL, server_token = NULL')
        except:
            logger.error('Failed to remove tokens from downloaded database.')
            cherrypy.response.status = 500
            return 'Error downloading database. Check the logs.'
        return serve_download(database_copy, name=database_file)

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    @addtoapi()
    def download_log(self, logfile='', **kwargs):
        if False:
            while True:
                i = 10
        ' Download the Tautulli log file.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                logfile (str):          The name of the Tautulli log file,\n                                        "tautulli", "tautulli_api", "plex_websocket"\n\n            Returns:\n                download\n            ```\n        '
        if logfile == 'tautulli_api':
            filename = logger.FILENAME_API
            log = logger.logger_api
        elif logfile == 'plex_websocket':
            filename = logger.FILENAME_PLEX_WEBSOCKET
            log = logger.logger_plex_websocket
        else:
            filename = logger.FILENAME
            log = logger.logger
        try:
            log.flush()
        except:
            pass
        return serve_download(os.path.join(plexpy.CONFIG.LOG_DIR, filename), name=filename)

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    @addtoapi()
    def download_plex_log(self, logfile='', **kwargs):
        if False:
            while True:
                i = 10
        ' Download the Plex log file.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                logfile (int):          The name of the Plex log file,\n                                        e.g. "Plex Media Server", "Plex Media Scanner"\n\n            Returns:\n                download\n            ```\n        '
        if not plexpy.CONFIG.PMS_LOGS_FOLDER:
            return 'Plex log folder not set in the settings.'
        if kwargs.get('log_type'):
            logfile = 'Plex Media ' + kwargs['log_type'].capitalize()
        log_file = (logfile or 'Plex Media Server') + '.log'
        log_file_path = os.path.join(plexpy.CONFIG.PMS_LOGS_FOLDER, log_file)
        if log_file and os.path.isfile(log_file_path):
            log_file_name = os.path.basename(log_file_path)
            return serve_download(log_file_path, name=log_file_name)
        else:
            return "Plex log file '%s' not found." % log_file

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def delete_image_cache(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Delete and recreate the image cache directory. '
        return self.delete_cache(folder='images')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def delete_cache(self, folder='', **kwargs):
        if False:
            print('Hello World!')
        ' Delete and recreate the cache directory. '
        cache_dir = os.path.join(plexpy.CONFIG.CACHE_DIR, folder)
        result = 'success'
        msg = 'Cleared the %scache.' % (folder + ' ' if folder else '')
        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
        except OSError as e:
            result = 'error'
            msg = 'Failed to delete %s.' % cache_dir
            logger.exception('Failed to delete %s: %s.' % (cache_dir, e))
            return {'result': result, 'message': msg}
        try:
            os.makedirs(cache_dir)
        except OSError as e:
            result = 'error'
            msg = 'Failed to make %s.' % cache_dir
            logger.exception('Failed to create %s: %s.' % (cache_dir, e))
            return {'result': result, 'message': msg}
        logger.info(msg)
        return {'result': result, 'message': msg}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def delete_hosted_images(self, rating_key='', service='', delete_all=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Delete the images uploaded to image hosting services.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                rating_key (int):       1234\n                                        (Note: Must be the movie, show, season, artist, or album rating key)\n                service (str):          \'imgur\' or \'cloudinary\'\n                delete_all (bool):      \'true\' to delete all images form the service\n\n            Returns:\n                json:\n                    {"result": "success",\n                     "message": "Deleted hosted images from Imgur."}\n            ```\n        '
        delete_all = helpers.bool_true(delete_all)
        data_factory = datafactory.DataFactory()
        result = data_factory.delete_img_info(rating_key=rating_key, service=service, delete_all=delete_all)
        if result:
            return {'result': 'success', 'message': 'Deleted hosted images from %s.' % result.capitalize()}
        else:
            return {'result': 'error', 'message': 'Failed to delete hosted images.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def delete_lookup_info(self, rating_key='', service='', delete_all=False, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Delete the 3rd party API lookup info.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                rating_key (int):       1234\n                                        (Note: Must be the movie, show, artist, album, or track rating key)\n                service (str):          \'themoviedb\' or \'tvmaze\' or \'musicbrainz\'\n                delete_all (bool):      \'true\' to delete all images form the service\n\n            Returns:\n                json:\n                    {"result": "success",\n                     "message": "Deleted lookup info."}\n            ```\n        '
        data_factory = datafactory.DataFactory()
        result = data_factory.delete_lookup_info(rating_key=rating_key, service=service, delete_all=delete_all)
        if result:
            return {'result': 'success', 'message': 'Deleted lookup info.'}
        else:
            return {'result': 'error', 'message': 'Failed to delete lookup info.'}

    @cherrypy.expose
    @requireAuth()
    def search(self, query='', **kwargs):
        if False:
            print('Hello World!')
        return serve_template(template_name='search.html', title='Search', query=query)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi('search')
    def search_results(self, query='', limit='', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Get search results from the PMS.\n\n            ```\n            Required parameters:\n                query (str):        The query string to search for\n\n            Optional parameters:\n                limit (int):        The maximum number of items to return per media type\n\n            Returns:\n                json:\n                    {"results_count": 69,\n                     "results_list":\n                        {"movie":\n                            [{...},\n                             {...},\n                             ]\n                         },\n                        {"episode":\n                            [{...},\n                             {...},\n                             ]\n                         },\n                        {...}\n                     }\n            ```\n        '
        pms_connect = pmsconnect.PmsConnect()
        result = pms_connect.get_search_results(query=query, limit=limit)
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for search_results.')
            return result

    @cherrypy.expose
    @requireAuth()
    def get_search_results_children(self, query='', limit='', media_type=None, season_index=None, **kwargs):
        if False:
            print('Hello World!')
        pms_connect = pmsconnect.PmsConnect()
        result = pms_connect.get_search_results(query=query, limit=limit)
        if media_type:
            result['results_list'] = {media_type: result['results_list'][media_type]}
        if media_type == 'season' and season_index:
            result['results_list']['season'] = [season for season in result['results_list']['season'] if season['media_index'] == season_index]
        if result:
            return serve_template(template_name='info_search_results_list.html', data=result, title='Search Result List')
        else:
            logger.warn('Unable to retrieve data for get_search_results_children.')
            return serve_template(template_name='info_search_results_list.html', data=None, title='Search Result List')

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def update_metadata(self, rating_key=None, query=None, update=False, **kwargs):
        if False:
            i = 10
            return i + 15
        query_string = query
        update = helpers.bool_true(update)
        data_factory = datafactory.DataFactory()
        query = data_factory.get_search_query(rating_key=rating_key)
        if query and query_string:
            query['query_string'] = query_string
        if query:
            return serve_template(template_name='update_metadata.html', query=query, update=update, title='Info')
        else:
            logger.warn('Unable to retrieve data for update_metadata.')
            return serve_template(template_name='update_metadata.html', query=query, update=update, title='Info')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def update_metadata_details(self, old_rating_key, new_rating_key, media_type, single_update=False, **kwargs):
        if False:
            print('Hello World!')
        ' Update the metadata in the Tautulli database by matching rating keys.\n            Also updates all parents or children of the media item if it is a show/season/episode\n            or artist/album/track.\n\n            ```\n            Required parameters:\n                old_rating_key (str):       12345\n                new_rating_key (str):       54321\n                media_type (str):           "movie", "show", "season", "episode", "artist", "album", "track"\n\n            Optional parameters:\n                None\n\n            Returns:\n                None\n            ```\n        '
        single_update = helpers.bool_true(single_update)
        if new_rating_key:
            data_factory = datafactory.DataFactory()
            pms_connect = pmsconnect.PmsConnect()
            old_key_list = data_factory.get_rating_keys_list(rating_key=old_rating_key, media_type=media_type)
            new_key_list = pms_connect.get_rating_keys_list(rating_key=new_rating_key, media_type=media_type)
            result = data_factory.update_metadata(old_key_list=old_key_list, new_key_list=new_key_list, media_type=media_type, single_update=single_update)
        if result:
            return {'message': result}
        else:
            return {'message': 'no data received'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_new_rating_keys(self, rating_key='', media_type='', **kwargs):
        if False:
            return 10
        ' Get a list of new rating keys for the PMS of all of the item\'s parent/children.\n\n            ```\n            Required parameters:\n                rating_key (str):       \'12345\'\n                media_type (str):       "movie", "show", "season", "episode", "artist", "album", "track"\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    {}\n            ```\n        '
        pms_connect = pmsconnect.PmsConnect()
        result = pms_connect.get_rating_keys_list(rating_key=rating_key, media_type=media_type)
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_new_rating_keys.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_old_rating_keys(self, rating_key='', media_type='', **kwargs):
        if False:
            while True:
                i = 10
        ' Get a list of old rating keys from the Tautulli database for all of the item\'s parent/children.\n\n            ```\n            Required parameters:\n                rating_key (str):       \'12345\'\n                media_type (str):       "movie", "show", "season", "episode", "artist", "album", "track"\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    {}\n            ```\n        '
        data_factory = datafactory.DataFactory()
        result = data_factory.get_rating_keys_list(rating_key=rating_key, media_type=media_type)
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_old_rating_keys.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def get_pms_sessions_json(self, **kwargs):
        if False:
            print('Hello World!')
        ' Get all the current sessions. '
        pms_connect = pmsconnect.PmsConnect()
        result = pms_connect.get_sessions('json')
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_pms_sessions_json.')
            return False

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi('get_metadata')
    def get_metadata_details(self, rating_key='', sync_id='', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Get the metadata for a media item.\n\n            ```\n            Required parameters:\n                rating_key (str):       Rating key of the item, OR\n                sync_id (str):          Sync ID of a synced item\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    {"actors": [\n                        "Emilia Clarke",\n                        "Lena Headey",\n                        "Sophie Turner",\n                        "Kit Harington",\n                        "Peter Dinklage",\n                        "Nikolaj Coster-Waldau",\n                        "Maisie Williams",\n                        "Iain Glen",\n                        "John Bradley",\n                        "Alfie Allen"\n                     ],\n                     "added_at": "1461572396",\n                     "art": "/library/metadata/1219/art/1462175063",\n                     "audience_rating": "7.4",\n                     "audience_rating_image": "themoviedb://image.rating",\n                     "banner": "/library/metadata/1219/banner/1462175063",\n                     "collections": [],\n                     "content_rating": "TV-MA",\n                     "directors": [\n                        "Jeremy Podeswa"\n                     ],\n                     "duration": "2998290",\n                     "edition_title": "",\n                     "full_title": "Game of Thrones - The Red Woman",\n                     "genres": [\n                        "Action/Adventure",\n                        "Drama",\n                        "Fantasy",\n                        "Romance"\n                     ],\n                     "grandparent_guid": "plex://show/5d9c086c46115600200aa2fe",\n                     "grandparent_guids": [\n                         "imdb://tt0944947",\n                         "tmdb://1399",\n                         "tvdb://121361"\n                     ],\n                     "grandparent_rating_key": "1219",\n                     "grandparent_thumb": "/library/metadata/1219/thumb/1462175063",\n                     "grandparent_title": "Game of Thrones",\n                     "grandparent_year": "2011",\n                     "guid": "plex://episode/5d9c1276e9d5a1001f4ff2fa",\n                     "guids": [\n                         "imdb://tt3658014",\n                         "tmdb://1156503",\n                         "tvdb://5469015"\n                     ],\n                     "labels": [],\n                     "last_viewed_at": "1462165717",\n                     "library_name": "TV Shows",\n                     "live": 0,\n                     "markers": [\n                        {\n                             "id": 908,\n                             "type": "credits",\n                             "start_time_offset": 2923863,\n                             "end_time_offset": 2998197,\n                             "first": true,\n                             "final": true\n                        },\n                        {\n                             "id": 908,\n                             "type": "intro",\n                             "start_time_offset": 1622,\n                             "end_time_offset": 109135,\n                             "first": null,\n                             "final": null\n                        }\n                     ],\n                     "media_index": "1",\n                     "media_info": [\n                         {\n                             "aspect_ratio": "1.78",\n                             "audio_channel_layout": "5.1",\n                             "audio_channels": "6",\n                             "audio_codec": "ac3",\n                             "audio_profile": "",\n                             "bitrate": "10617",\n                             "channel_call_sign": "",\n                             "channel_identifier": "",\n                             "channel_thumb": "",\n                             "container": "mkv",\n                             "height": "1078",\n                             "id": "257925",\n                             "optimized_version": 0,\n                             "parts": [\n                                 {\n                                     "file": "/media/TV Shows/Game of Thrones/Season 06/Game of Thrones - S06E01 - The Red Woman.mkv",\n                                     "file_size": "3979115377",\n                                     "id": "274169",\n                                     "indexes": 1,\n                                     "streams": [\n                                         {\n                                             "id": "511663",\n                                             "type": "1",\n                                             "video_bit_depth": "8",\n                                             "video_bitrate": "10233",\n                                             "video_codec": "h264",\n                                             "video_codec_level": "41",\n                                             "video_color_primaries": "",\n                                             "video_color_range": "tv",\n                                             "video_color_space": "bt709",\n                                             "video_color_trc": "",\n                                             "video_dynamic_range": "SDR",\n                                             "video_frame_rate": "23.976",\n                                             "video_height": "1078",\n                                             "video_language": "",\n                                             "video_language_code": "",\n                                             "video_profile": "high",\n                                             "video_ref_frames": "4",\n                                             "video_scan_type": "progressive",\n                                             "video_width": "1920",\n                                             "selected": 0\n                                         },\n                                         {\n                                             "audio_bitrate": "384",\n                                             "audio_bitrate_mode": "",\n                                             "audio_channel_layout": "5.1(side)",\n                                             "audio_channels": "6",\n                                             "audio_codec": "ac3",\n                                             "audio_language": "",\n                                             "audio_language_code": "",\n                                             "audio_profile": "",\n                                             "audio_sample_rate": "48000",\n                                             "id": "511664",\n                                             "type": "2",\n                                             "selected": 1\n                                         },\n                                         {\n                                             "id": "511953",\n                                             "subtitle_codec": "srt",\n                                             "subtitle_container": "",\n                                             "subtitle_forced": 0,\n                                             "subtitle_format": "srt",\n                                             "subtitle_language": "English",\n                                             "subtitle_language_code": "eng",\n                                             "subtitle_location": "external",\n                                             "type": "3",\n                                             "selected": 1\n                                         }\n                                     ]\n                                 }\n                             ],\n                             "video_codec": "h264",\n                             "video_framerate": "24p",\n                             "video_full_resolution": "1080p",\n                             "video_profile": "high",\n                             "video_resolution": "1080",\n                             "width": "1920"\n                         }\n                     ],\n                     "media_type": "episode",\n                     "original_title": "",\n                     "originally_available_at": "2016-04-24",\n                     "parent_guid": "plex://season/602e67e61d3358002c4120f7",\n                     "parent_guids": [\n                         "tvdb://651357"\n                     ],\n                     "parent_media_index": "6",\n                     "parent_rating_key": "153036",\n                     "parent_thumb": "/library/metadata/153036/thumb/1462175062",\n                     "parent_title": "Season 6",\n                     "parent_year": "2016",\n                     "rating": "",\n                     "rating_image": "",\n                     "rating_key": "153037",\n                     "section_id": "2",\n                     "sort_title": "Red Woman",\n                     "studio": "Revolution Sun Studios",\n                     "summary": "The fate of Jon Snow is revealed. Daenerys meets a strong man. Cersei sees her daughter once again.",\n                     "tagline": "",\n                     "thumb": "/library/metadata/153037/thumb/1462175060",\n                     "title": "The Red Woman",\n                     "updated_at": "1462175060",\n                     "user_rating": "9.0",\n                     "writers": [\n                        "David Benioff",\n                        "D. B. Weiss"\n                     ],\n                     "year": "2016"\n                     }\n            ```\n        '
        pms_connect = pmsconnect.PmsConnect()
        metadata = pms_connect.get_metadata_details(rating_key=rating_key, sync_id=sync_id)
        if metadata:
            return metadata
        else:
            logger.warn('Unable to retrieve data for get_metadata_details.')
            return metadata

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi('get_recently_added')
    def get_recently_added_details(self, start='0', count='0', media_type='', section_id='', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Get all items that where recently added to plex.\n\n            ```\n            Required parameters:\n                count (str):        Number of items to return\n\n            Optional parameters:\n                start (str):        The item number to start at\n                media_type (str):   The media type: movie, show, artist\n                section_id (str):   The id of the Plex library section\n\n            Returns:\n                json:\n                    {"recently_added":\n                        [{"actors": [\n                             "Kit Harington",\n                             "Emilia Clarke",\n                             "Isaac Hempstead-Wright",\n                             "Maisie Williams",\n                             "Liam Cunningham",\n                          ],\n                          "added_at": "1461572396",\n                          "art": "/library/metadata/1219/art/1462175063",\n                          "audience_rating": "8",\n                          "audience_rating_image": "rottentomatoes://image.rating.upright",\n                          "banner": "/library/metadata/1219/banner/1462175063",\n                          "directors": [\n                             "Jeremy Podeswa"\n                          ],\n                          "duration": "2998290",\n                          "full_title": "Game of Thrones - The Red Woman",\n                          "genres": [\n                             "Adventure",\n                             "Drama",\n                             "Fantasy"\n                          ],\n                          "grandparent_rating_key": "1219",\n                          "grandparent_thumb": "/library/metadata/1219/thumb/1462175063",\n                          "grandparent_title": "Game of Thrones",\n                          "guid": "com.plexapp.agents.thetvdb://121361/6/1?lang=en",\n                          "guids": [],\n                          "labels": [],\n                          "last_viewed_at": "1462165717",\n                          "library_name": "TV Shows",\n                          "media_index": "1",\n                          "media_type": "episode",\n                          "original_title": "",\n                          "originally_available_at": "2016-04-24",\n                          "parent_media_index": "6",\n                          "parent_rating_key": "153036",\n                          "parent_thumb": "/library/metadata/153036/thumb/1462175062",\n                          "parent_title": "",\n                          "rating": "7.8",\n                          "rating_image": "rottentomatoes://image.rating.ripe",\n                          "rating_key": "153037",\n                          "section_id": "2",\n                          "sort_title": "Red Woman",\n                          "studio": "HBO",\n                          "summary": "Jon Snow is dead. Daenerys meets a strong man. Cersei sees her daughter again.",\n                          "tagline": "",\n                          "thumb": "/library/metadata/153037/thumb/1462175060",\n                          "title": "The Red Woman",\n                          "user_rating": "9.0",\n                          "updated_at": "1462175060",\n                          "writers": [\n                             "David Benioff",\n                             "D. B. Weiss"\n                          ],\n                          "year": "2016"\n                          },\n                         {...},\n                         {...}\n                         ]\n                     }\n            ```\n        '
        if 'type' in kwargs:
            media_type = kwargs['type']
        pms_connect = pmsconnect.PmsConnect()
        result = pms_connect.get_recently_added_details(start=start, count=count, media_type=media_type, section_id=section_id)
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_recently_added_details.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def get_friends_list(self, **kwargs):
        if False:
            return 10
        ' Get the friends list of the server owner for Plex.tv. '
        plex_tv = plextv.PlexTV()
        result = plex_tv.get_plextv_friends('json')
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_friends_list.')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def get_user_details(self, **kwargs):
        if False:
            print('Hello World!')
        " Get all details about a the server's owner from Plex.tv. "
        plex_tv = plextv.PlexTV()
        result = plex_tv.get_plextv_user_details('json')
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_user_details.')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def get_server_list(self, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Find all servers published on Plex.tv '
        plex_tv = plextv.PlexTV()
        result = plex_tv.get_plextv_server_list('json')
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_server_list.')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def get_sync_lists(self, machine_id='', **kwargs):
        if False:
            return 10
        ' Get all items that are currently synced from the PMS. '
        plex_tv = plextv.PlexTV()
        result = plex_tv.get_plextv_sync_lists(machine_id=machine_id, output_format='json')
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_sync_lists.')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def get_servers(self, **kwargs):
        if False:
            print('Hello World!')
        pms_connect = pmsconnect.PmsConnect()
        result = pms_connect.get_server_list(output_format='json')
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_servers.')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_servers_info(self, **kwargs):
        if False:
            print('Hello World!')
        ' Get info about the PMS.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    [{"port": "32400",\n                      "host": "10.0.0.97",\n                      "version": "0.9.15.2.1663-7efd046",\n                      "name": "Winterfell-Server",\n                      "machine_identifier": "ds48g4r354a8v9byrrtr697g3g79w"\n                      }\n                     ]\n            ```\n        '
        pms_connect = pmsconnect.PmsConnect()
        result = pms_connect.get_servers_info()
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_servers_info.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_server_identity(self, **kwargs):
        if False:
            return 10
        ' Get info about the local server.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    [{"machine_identifier": "ds48g4r354a8v9byrrtr697g3g79w",\n                      "version": "0.9.15.x.xxx-xxxxxxx"\n                      }\n                     ]\n            ```\n        '
        pms_connect = pmsconnect.PmsConnect()
        result = pms_connect.get_server_identity()
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_server_identity.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_server_friendly_name(self, **kwargs):
        if False:
            print('Hello World!')
        ' Get the name of the PMS.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                None\n\n            Returns:\n                string:     "Winterfell-Server"\n            ```\n        '
        result = pmsconnect.get_server_friendly_name()
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_server_friendly_name.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @addtoapi()
    def get_activity(self, session_key=None, session_id=None, **kwargs):
        if False:
            while True:
                i = 10
        ' Get the current activity on the PMS.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                session_key (int):    Session key for the session info to return, OR\n                session_id (str):     Session ID for the session info to return\n\n            Returns:\n                json:\n                    {"lan_bandwidth": 25318,\n                     "sessions": [\n                         {\n                             "actors": [\n                                 "Kit Harington",\n                                 "Emilia Clarke",\n                                 "Isaac Hempstead-Wright",\n                                 "Maisie Williams",\n                                 "Liam Cunningham",\n                             ],\n                             "added_at": "1461572396",\n                             "allow_guest": 1,\n                             "art": "/library/metadata/1219/art/1503306930",\n                             "aspect_ratio": "1.78",\n                             "audience_rating": "",\n                             "audience_rating_image": "rottentomatoes://image.rating.upright",\n                             "audio_bitrate": "384",\n                             "audio_bitrate_mode": "",\n                             "audio_channel_layout": "5.1(side)",\n                             "audio_channels": "6",\n                             "audio_codec": "ac3",\n                             "audio_decision": "direct play",\n                             "audio_language": "",\n                             "audio_language_code": "",\n                             "audio_profile": "",\n                             "audio_sample_rate": "48000",\n                             "bandwidth": "25318",\n                             "banner": "/library/metadata/1219/banner/1503306930",\n                             "bif_thumb": "/library/parts/274169/indexes/sd/1000",\n                             "bitrate": "10617",\n                             "channel_call_sign": "",\n                             "channel_identifier": "",\n                             "channel_stream": 0,\n                             "channel_thumb": "",\n                             "children_count": "",\n                             "collections": [],\n                             "container": "mkv",\n                             "container_decision": "direct play",\n                             "content_rating": "TV-MA",\n                             "deleted_user": 0,\n                             "device": "Windows",\n                             "directors": [\n                                 "Jeremy Podeswa"\n                             ],\n                             "do_notify": 0,\n                             "duration": "2998272",\n                             "email": "Jon.Snow.1337@CastleBlack.com",\n                             "file": "/media/TV Shows/Game of Thrones/Season 06/Game of Thrones - S06E01 - The Red Woman.mkv",\n                             "file_size": "3979115377",\n                             "friendly_name": "Jon Snow",\n                             "full_title": "Game of Thrones - The Red Woman",\n                             "genres": [\n                                 "Adventure",\n                                 "Drama",\n                                 "Fantasy"\n                             ],\n                             "grandparent_guid": "com.plexapp.agents.thetvdb://121361?lang=en",\n                             "grandparent_rating_key": "1219",\n                             "grandparent_thumb": "/library/metadata/1219/thumb/1503306930",\n                             "grandparent_title": "Game of Thrones",\n                             "guid": "com.plexapp.agents.thetvdb://121361/6/1?lang=en",\n                             "height": "1078",\n                             "id": "",\n                             "indexes": 1,\n                             "ip_address": "10.10.10.1",\n                             "ip_address_public": "64.123.23.111",\n                             "is_admin": 1,\n                             "is_allow_sync": 1,\n                             "is_home_user": 1,\n                             "is_restricted": 0,\n                             "keep_history": 1,\n                             "labels": [],\n                             "last_viewed_at": "1462165717",\n                             "library_name": "TV Shows",\n                             "live": 0,\n                             "live_uuid": "",\n                             "local": "1",\n                             "location": "lan",\n                             "machine_id": "lmd93nkn12k29j2lnm",\n                             "media_index": "1",\n                             "media_type": "episode",\n                             "optimized_version": 0,\n                             "optimized_version_profile": "",\n                             "optimized_version_title": "",\n                             "original_title": "",\n                             "originally_available_at": "2016-04-24",\n                             "parent_guid": "com.plexapp.agents.thetvdb://121361/6?lang=en",\n                             "parent_media_index": "6",\n                             "parent_rating_key": "153036",\n                             "parent_thumb": "/library/metadata/153036/thumb/1503889210",\n                             "parent_title": "Season 6",\n                             "platform": "Plex Media Player",\n                             "platform_name": "plex",\n                             "platform_version": "2.4.1.787-54a020cd",\n                             "player": "Castle-PC",\n                             "product": "Plex Media Player",\n                             "product_version": "3.35.2",\n                             "profile": "Konvergo",\n                             "progress_percent": "0",\n                             "quality_profile": "Original",\n                             "rating": "7.8",\n                             "rating_image": "rottentomatoes://image.rating.ripe",\n                             "rating_key": "153037",\n                             "relay": 0,\n                             "section_id": "2",\n                             "secure": 1,\n                             "session_id": "helf15l3rxgw01xxe0jf3l3d",\n                             "session_key": "27",\n                             "shared_libraries": [\n                                 "10",\n                                 "1",\n                                 "4",\n                                 "5",\n                                 "15",\n                                 "20",\n                                 "2"\n                             ],\n                             "sort_title": "Red Woman",\n                             "state": "playing",\n                             "stream_aspect_ratio": "1.78",\n                             "stream_audio_bitrate": "384",\n                             "stream_audio_bitrate_mode": "",\n                             "stream_audio_channel_layout": "5.1(side)",\n                             "stream_audio_channel_layout_": "5.1(side)",\n                             "stream_audio_channels": "6",\n                             "stream_audio_codec": "ac3",\n                             "stream_audio_decision": "direct play",\n                             "stream_audio_language": "",\n                             "stream_audio_language_code": "",\n                             "stream_audio_sample_rate": "48000",\n                             "stream_bitrate": "10617",\n                             "stream_container": "mkv",\n                             "stream_container_decision": "direct play",\n                             "stream_duration": "2998272",\n                             "stream_subtitle_codec": "",\n                             "stream_subtitle_container": "",\n                             "stream_subtitle_decision": "",\n                             "stream_subtitle_forced": 0,\n                             "stream_subtitle_format": "",\n                             "stream_subtitle_language": "",\n                             "stream_subtitle_language_code": "",\n                             "stream_subtitle_location": "",\n                             "stream_video_bit_depth": "8",\n                             "stream_video_bitrate": "10233",\n                             "stream_video_chroma_subsampling": "4:2:0",\n                             "stream_video_codec": "h264",\n                             "stream_video_codec_level": "41",\n                             "stream_video_color_primaries": "",\n                             "stream_video_color_range": "tv",\n                             "stream_video_color_space": "bt709",\n                             "stream_video_color_trc": "",\n                             "stream_video_decision": "direct play",\n                             "stream_video_dynamic_range": "SDR",\n                             "stream_video_framerate": "24p",\n                             "stream_video_full_resolution": "1080p",\n                             "stream_video_height": "1078",\n                             "stream_video_language": "",\n                             "stream_video_language_code": "",\n                             "stream_video_ref_frames": "4",\n                             "stream_video_resolution": "1080",\n                             "stream_video_scan_type": "progressive",\n                             "stream_video_width": "1920",\n                             "studio": "HBO",\n                             "subtitle_codec": "",\n                             "subtitle_container": "",\n                             "subtitle_decision": "",\n                             "subtitle_forced": 0,\n                             "subtitle_format": "",\n                             "subtitle_language": "",\n                             "subtitle_language_code": "",\n                             "subtitle_location": "",\n                             "subtitles": 0,\n                             "summary": "Jon Snow is dead. Daenerys meets a strong man. Cersei sees her daughter again.",\n                             "synced_version": 0,\n                             "synced_version_profile": "",\n                             "tagline": "",\n                             "throttled": "0",\n                             "thumb": "/library/metadata/153037/thumb/1503889207",\n                             "title": "The Red Woman",\n                             "transcode_audio_channels": "",\n                             "transcode_audio_codec": "",\n                             "transcode_container": "",\n                             "transcode_decision": "direct play",\n                             "transcode_height": "",\n                             "transcode_hw_decode": "",\n                             "transcode_hw_decode_title": "",\n                             "transcode_hw_decoding": 0,\n                             "transcode_hw_encode": "",\n                             "transcode_hw_encode_title": "",\n                             "transcode_hw_encoding": 0,\n                             "transcode_hw_full_pipeline": 0,\n                             "transcode_hw_requested": 0,\n                             "transcode_key": "",\n                             "transcode_max_offset_available": 0,\n                             "transcode_min_offset_available": 0,\n                             "transcode_progress": 0,\n                             "transcode_protocol": "",\n                             "transcode_speed": "",\n                             "transcode_throttled": 0,\n                             "transcode_video_codec": "",\n                             "transcode_width": "",\n                             "type": "",\n                             "updated_at": "1503889207",\n                             "user": "LordCommanderSnow",\n                             "user_id": 133788,\n                             "user_rating": "",\n                             "user_thumb": "https://plex.tv/users/k10w42309cynaopq/avatar",\n                             "username": "LordCommanderSnow",\n                             "video_bit_depth": "8",\n                             "video_bitrate": "10233",\n                             "video_chroma_subsampling": "4:2:0",\n                             "video_codec": "h264",\n                             "video_codec_level": "41",\n                             "video_color_primaries": "",\n                             "video_color_range": "tv",\n                             "video_color_space": "bt709",\n                             "video_color_trc": ",\n                             "video_decision": "direct play",\n                             "video_dynamic_range": "SDR",\n                             "video_frame_rate": "23.976",\n                             "video_framerate": "24p",\n                             "video_full_resolution": "1080p",\n                             "video_height": "1078",\n                             "video_language": "",\n                             "video_language_code": "",\n                             "video_profile": "high",\n                             "video_ref_frames": "4",\n                             "video_resolution": "1080",\n                             "video_scan_type": "progressive",\n                             "video_width": "1920",\n                             "view_offset": "1000",\n                             "width": "1920",\n                             "writers": [\n                                 "David Benioff",\n                                 "D. B. Weiss"\n                             ],\n                             "year": "2016"\n                         }\n                     ],\n                     "stream_count": "1",\n                     "stream_count_direct_play": 1,\n                     "stream_count_direct_stream": 0,\n                     "stream_count_transcode": 0,\n                     "total_bandwidth": 25318,\n                     "wan_bandwidth": 0\n                     }\n            ```\n        '
        try:
            pms_connect = pmsconnect.PmsConnect(token=plexpy.CONFIG.PMS_TOKEN)
            result = pms_connect.get_current_activity()
            if result:
                if session_key:
                    return next((s for s in result['sessions'] if s['session_key'] == session_key), {})
                if session_id:
                    return next((s for s in result['sessions'] if s['session_id'] == session_id), {})
                counts = {'stream_count_direct_play': 0, 'stream_count_direct_stream': 0, 'stream_count_transcode': 0, 'total_bandwidth': 0, 'lan_bandwidth': 0, 'wan_bandwidth': 0}
                for s in result['sessions']:
                    if s['transcode_decision'] == 'transcode':
                        counts['stream_count_transcode'] += 1
                    elif s['transcode_decision'] == 'copy':
                        counts['stream_count_direct_stream'] += 1
                    else:
                        counts['stream_count_direct_play'] += 1
                    counts['total_bandwidth'] += helpers.cast_to_int(s['bandwidth'])
                    if s['location'] == 'lan':
                        counts['lan_bandwidth'] += helpers.cast_to_int(s['bandwidth'])
                    else:
                        counts['wan_bandwidth'] += helpers.cast_to_int(s['bandwidth'])
                result.update(counts)
                return result
            else:
                logger.warn('Unable to retrieve data for get_activity.')
                return {}
        except Exception as e:
            logger.exception('Unable to retrieve data for get_activity: %s' % e)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi('get_libraries')
    def get_full_libraries_list(self, **kwargs):
        if False:
            print('Hello World!')
        ' Get a list of all libraries on your server.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    [{"art": "/:/resources/show-fanart.jpg",\n                      "child_count": "3745",\n                      "count": "62",\n                      "is_active": 1,\n                      "parent_count": "240",\n                      "section_id": "2",\n                      "section_name": "TV Shows",\n                      "section_type": "show",\n                      "thumb": "/:/resources/show.png"\n                      },\n                     {...},\n                     {...}\n                     ]\n            ```\n        '
        pms_connect = pmsconnect.PmsConnect()
        result = pms_connect.get_library_details()
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_full_libraries_list.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi('get_users')
    def get_full_users_list(self, **kwargs):
        if False:
            return 10
        ' Get a list of all users that have access to your server.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    [{"allow_guest": 1,\n                      "do_notify": 1,\n                      "email": "Jon.Snow.1337@CastleBlack.com",\n                      "filter_all": "",\n                      "filter_movies": "",\n                      "filter_music": "",\n                      "filter_photos": "",\n                      "filter_tv": "",\n                      "is_active": 1,\n                      "is_admin": 0,\n                      "is_allow_sync": 1,\n                      "is_home_user": 1,\n                      "is_restricted": 0,\n                      "keep_history": 1,\n                      "row_id": 1,\n                      "shared_libraries": ["1", "2", "3"],\n                      "thumb": "https://plex.tv/users/k10w42309cynaopq/avatar",\n                      "user_id": "133788",\n                      "username": "Jon Snow"\n                      },\n                     {...},\n                     {...}\n                     ]\n            ```\n        '
        user_data = users.Users()
        result = user_data.get_users()
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_full_users_list.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @sanitize_out()
    @addtoapi()
    def get_synced_items(self, machine_id='', user_id='', **kwargs):
        if False:
            print('Hello World!')
        ' Get a list of synced items on the PMS.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                machine_id (str):       The PMS identifier\n                user_id (str):          The id of the Plex user\n\n            Returns:\n                json:\n                    [{"audio_bitrate": "192",\n                      "client_id": "95434se643fsf24f-com-plexapp-android",\n                      "content_type": "video",\n                      "device_name": "Tyrion\'s iPad",\n                      "failure": "",\n                      "item_complete_count": "1",\n                      "item_count": "1",\n                      "item_downloaded_count": "1",\n                      "item_downloaded_percent_complete": 100,\n                      "metadata_type": "movie",\n                      "photo_quality": "74",\n                      "platform": "iOS",\n                      "rating_key": "154092",\n                      "root_title": "Movies",\n                      "state": "complete",\n                      "sync_id": "11617019",\n                      "sync_media_type": null,\n                      "sync_title": "Deadpool",\n                      "total_size": "560718134",\n                      "user": "DrukenDwarfMan",\n                      "user_id": "696969",\n                      "username": "DrukenDwarfMan",\n                      "video_bitrate": "4000"\n                      "video_quality": "100"\n                      },\n                     {...},\n                     {...}\n                     ]\n            ```\n        '
        plex_tv = plextv.PlexTV()
        result = plex_tv.get_synced_items(machine_id=machine_id, user_id_filter=user_id)
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_synced_items.')
            return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def get_sync_transcode_queue(self, **kwargs):
        if False:
            return 10
        ' Return details for currently syncing items. '
        pms_connect = pmsconnect.PmsConnect()
        result = pms_connect.get_sync_transcode_queue(output_format='json')
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_sync_transcode_queue.')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_home_stats(self, grouping=None, time_range=30, stats_type='plays', stats_start=0, stats_count=10, stat_id='', section_id=None, user_id=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Get the homepage watch statistics.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                grouping (int):         0 or 1\n                time_range (int):       The time range to calculate statistics, 30\n                stats_type (str):       \'plays\' or \'duration\'\n                stats_start (int)       The row number of the stat item to start at, 0\n                stats_count (int):      The number of stat items to return, 5\n                stat_id (str):          A single stat to return, \'top_movies\', \'popular_movies\',\n                                        \'top_tv\', \'popular_tv\', \'top_music\', \'popular_music\', \'top_libraries\',\n                                        \'top_users\', \'top_platforms\', \'last_watched\', \'most_concurrent\'\n                section_id (int):       The id of the Plex library section\n                user_id (int):          The id of the Plex user\n\n            Returns:\n                json:\n                    [{"stat_id": "top_movies",\n                      "stat_type": "total_plays",\n                      "rows": [{...}]\n                      },\n                     {"stat_id": "popular_movies",\n                      "rows": [{...}]\n                      },\n                     {"stat_id": "top_tv",\n                      "stat_type": "total_plays",\n                      "rows":\n                        [{"content_rating": "TV-MA",\n                          "friendly_name": "",\n                          "grandparent_thumb": "/library/metadata/1219/thumb/1462175063",\n                          "guid": "com.plexapp.agents.thetvdb://121361/6/1?lang=en",\n                          "labels": [],\n                          "last_play": 1462380698,\n                          "live": 0,\n                          "media_type": "episode",\n                          "platform": "",\n                          "rating_key": 1219,\n                          "row_id": 1116,\n                          "section_id": 2,\n                          "thumb": "",\n                          "title": "Game of Thrones",\n                          "total_duration": 213302,\n                          "total_plays": 69,\n                          "user": "",\n                          "users_watched": ""\n                          },\n                         {...},\n                         {...}\n                         ]\n                      },\n                     {"stat_id": "popular_tv",\n                      "rows": [{...}]\n                      },\n                     {"stat_id": "top_music",\n                      "stat_type": "total_plays",\n                      "rows": [{...}]\n                      },\n                     {"stat_id": "popular_music",\n                      "rows": [{...}]\n                      },\n                     {"stat_id": "last_watched",\n                      "rows": [{...}]\n                      },\n                     {"stat_id": "top_libraries",\n                      "stat_type": "total_plays",\n                      "rows": [{...}]\n                      },\n                     {"stat_id": "top_users",\n                      "stat_type": "total_plays",\n                      "rows": [{...}]\n                      },\n                     {"stat_id": "top_platforms",\n                      "stat_type": "total_plays",\n                      "rows": [{...}]\n                      },\n                     {"stat_id": "most_concurrent",\n                      "rows": [{...}]\n                      }\n                     ]\n            ```\n        '
        if stats_type in (0, '0'):
            stats_type = 'plays'
        elif stats_type in (1, '1'):
            stats_type = 'duration'
        grouping = helpers.bool_true(grouping, return_none=True)
        data_factory = datafactory.DataFactory()
        result = data_factory.get_home_stats(grouping=grouping, time_range=time_range, stats_type=stats_type, stats_start=stats_start, stats_count=stats_count, stat_id=stat_id, section_id=section_id, user_id=user_id)
        if result:
            return result
        else:
            logger.warn('Unable to retrieve data for get_home_stats.')
            return result

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    @addtoapi('arnold')
    def random_arnold_quotes(self, **kwargs):
        if False:
            return 10
        ' Get to the chopper! '
        import random
        quote_list = ['To crush your enemies, see them driven before you, and to hear the lamentation of their women!', 'Your clothes, give them to me, now!', 'Do it!', 'If it bleeds, we can kill it.', 'See you at the party Richter!', 'Let off some steam, Bennett.', "I'll be back.", 'Get to the chopper!', 'Hasta La Vista, Baby!', "It's not a tumor!", 'Dillon, you son of a bitch!', 'Benny!! Screw you!!', 'Stop whining! You kids are soft. You lack discipline.', 'Nice night for a walk.', 'Stick around!', 'I need your clothes, your boots and your motorcycle.', "No, it's not a tumor. It's not a tumor!", 'I LIED!', 'Are you Sarah Connor?', "I'm a cop you idiot!", 'Come with me if you want to live.', 'Who is your daddy and what does he do?', "Oh, cookies! I can't wait to toss them.", 'Make it quick because my horse is getting tired.', 'What killed the dinosaurs? The Ice Age!', "That's for sleeping with my wife!", "Remember when I said I'd kill you last... I lied!", "You want to be a farmer? Here's a couple of acres.", 'Now, this is the plan. Get your ass to Mars.', 'I just had a terrible thought... What if this is a dream?', 'Well, listen to this one: Rubber baby buggy bumpers!', 'Take your toy back to the carpet!', 'My name is John Kimble... And I love my car.', 'I eat Green Berets for breakfast.', 'Put that cookie down! NOW!']
        return random.choice(quote_list)

    @cherrypy.expose
    def api(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if args and 'v2' in args[0]:
            return API2()._api_run(**kwargs)
        else:
            cherrypy.response.headers['Content-Type'] = 'application/json;charset=UTF-8'
            return json.dumps(API2()._api_responds(result_type='error', msg='Please use the /api/v2 endpoint.')).encode('utf-8')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_tautulli_info(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Get info about the Tautulli server.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    {"tautulli_install_type": "git",\n                     "tautulli_version": "v2.8.1",\n                     "tautulli_branch": "master",\n                     "tautulli_commit": "2410eb33805aaac4bd1c5dad0f71e4f15afaf742",\n                     "tautulli_platform": "Windows",\n                     "tautulli_platform_release": "10",\n                     "tautulli_platform_version": "10.0.19043",\n                     "tautulli_platform_linux_distro": "",\n                     "tautulli_platform_device_name": "Winterfell-Server",\n                     "tautulli_python_version": "3.10.0"\n                     }\n            ```\n        '
        return plexpy.get_tautulli_info()

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_pms_update(self, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Check for updates to the Plex Media Server.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    {"update_available": true,\n                     "platform": "Windows",\n                     "release_date": "1473721409",\n                     "version": "1.1.4.2757-24ffd60",\n                     "requirements": "...",\n                     "extra_info": "...",\n                     "changelog_added": "...",\n                     "changelog_fixed": "...",\n                     "label": "Download",\n                     "distro": "english",\n                     "distro_build": "windows-i386",\n                     "download_url": "https://downloads.plex.tv/...",\n                     }\n            ```\n        '
        plex_tv = plextv.PlexTV()
        result = plex_tv.get_plex_update()
        return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @addtoapi()
    def get_geoip_lookup(self, ip_address='', **kwargs):
        if False:
            print('Hello World!')
        ' Get the geolocation info for an IP address.\n\n            ```\n            Required parameters:\n                ip_address\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    {"city": "Mountain View",\n                     "code": "US",\n                     "continent": "NA",\n                     "country": "United States",\n                     "latitude": 37.386,\n                     "longitude": -122.0838,\n                     "postal_code": "94035",\n                     "region": "California",\n                     "timezone": "America/Los_Angeles",\n                     "accuracy": null\n                     }\n            ```\n        '
        message = ''
        if not ip_address:
            message = 'No IP address provided.'
        elif not helpers.is_valid_ip(ip_address):
            message = 'Invalid IP address provided: %s' % ip_address
        if message:
            return {'result': 'error', 'message': message}
        plex_tv = plextv.PlexTV()
        geo_info = plex_tv.get_geoip_lookup(ip_address)
        if geo_info:
            return {'result': 'success', 'data': geo_info}
        return {'result': 'error', 'message': 'Failed to lookup GeoIP info for address: %s' % ip_address}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth()
    @addtoapi()
    def get_whois_lookup(self, ip_address='', **kwargs):
        if False:
            while True:
                i = 10
        ' Get the connection info for an IP address.\n\n            ```\n            Required parameters:\n                ip_address\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    {"host": "google-public-dns-a.google.com",\n                     "nets": [{"description": "Google Inc.",\n                               "address": "1600 Amphitheatre Parkway",\n                               "city": "Mountain View",\n                               "state": "CA",\n                               "postal_code": "94043",\n                               "country": "United States",\n                               ...\n                               },\n                               {...}\n                              ]\n                json:\n                    {"host": "Not available",\n                     "nets": [],\n                     "error": "IPv4 address 127.0.0.1 is already defined as Loopback via RFC 1122, Section 3.2.1.3."\n                     }\n            ```\n        '
        whois_info = helpers.whois_lookup(ip_address)
        return whois_info

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def get_plexpy_url(self, **kwargs):
        if False:
            print('Hello World!')
        return helpers.get_plexpy_url()

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_newsletters(self, **kwargs):
        if False:
            return 10
        ' Get a list of configured newsletters.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    [{"id": 1,\n                      "agent_id": 0,\n                      "agent_name": "recently_added",\n                      "agent_label": "Recently Added",\n                      "friendly_name": "",\n                      "cron": "0 0 * * 1",\n                      "active": 1\n                      }\n                     ]\n            ```\n        '
        result = newsletters.get_newsletters()
        return result

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def get_newsletters_table(self, **kwargs):
        if False:
            i = 10
            return i + 15
        result = newsletters.get_newsletters()
        return serve_template(template_name='newsletters_table.html', newsletters_list=result)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def delete_newsletter(self, newsletter_id=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Remove a newsletter from the database.\n\n            ```\n            Required parameters:\n                newsletter_id (int):        The newsletter to delete\n\n            Optional parameters:\n                None\n\n            Returns:\n                None\n            ```\n        '
        result = newsletters.delete_newsletter(newsletter_id=newsletter_id)
        if result:
            return {'result': 'success', 'message': 'Newsletter deleted successfully.'}
        else:
            return {'result': 'error', 'message': 'Failed to delete newsletter.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_newsletter_config(self, newsletter_id=None, **kwargs):
        if False:
            while True:
                i = 10
        ' Get the configuration for an existing notification agent.\n\n            ```\n            Required parameters:\n                newsletter_id (int):        The newsletter config to retrieve\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    {"id": 1,\n                     "agent_id": 0,\n                     "agent_name": "recently_added",\n                     "agent_label": "Recently Added",\n                     "friendly_name": "",\n                     "id_name": "",\n                     "cron": "0 0 * * 1",\n                     "active": 1,\n                     "subject": "Recently Added to {server_name}! ({end_date})",\n                     "body": "View the newsletter here: {newsletter_url}",\n                     "message": "",\n                     "config": {"custom_cron": 0,\n                                "filename": "newsletter_{newsletter_uuid}.html",\n                                "formatted": 1,\n                                "incl_libraries": ["1", "2"],\n                                "notifier_id": 1,\n                                "save_only": 0,\n                                "time_frame": 7,\n                                "time_frame_units": "days"\n                                },\n                     "email_config": {...},\n                     "config_options": [{...}, ...],\n                     "email_config_options": [{...}, ...]\n                     }\n            ```\n        '
        result = newsletters.get_newsletter_config(newsletter_id=newsletter_id, mask_passwords=True)
        return result

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def get_newsletter_config_modal(self, newsletter_id=None, **kwargs):
        if False:
            i = 10
            return i + 15
        result = newsletters.get_newsletter_config(newsletter_id=newsletter_id, mask_passwords=True)
        return serve_template(template_name='newsletter_config.html', newsletter=result)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def add_newsletter_config(self, agent_id=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Add a new notification agent.\n\n            ```\n            Required parameters:\n                agent_id (int):           The newsletter type to add\n\n            Optional parameters:\n                None\n\n            Returns:\n                None\n            ```\n        '
        result = newsletters.add_newsletter_config(agent_id=agent_id, **kwargs)
        if result:
            return {'result': 'success', 'message': 'Added newsletter.', 'newsletter_id': result}
        else:
            return {'result': 'error', 'message': 'Failed to add newsletter.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def set_newsletter_config(self, newsletter_id=None, agent_id=None, **kwargs):
        if False:
            while True:
                i = 10
        " Configure an existing newsletter agent.\n\n            ```\n            Required parameters:\n                newsletter_id (int):    The newsletter config to update\n                agent_id (int):         The newsletter type of the newsletter\n\n            Optional parameters:\n                Pass all the config options for the agent with the 'newsletter_config_' and 'newsletter_email_' prefix.\n\n            Returns:\n                None\n            ```\n        "
        result = newsletters.set_newsletter_config(newsletter_id=newsletter_id, agent_id=agent_id, **kwargs)
        if result:
            return {'result': 'success', 'message': 'Saved newsletter.'}
        else:
            return {'result': 'error', 'message': 'Failed to save newsletter.'}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    def send_newsletter(self, newsletter_id=None, subject='', body='', message='', notify_action='', **kwargs):
        if False:
            print('Hello World!')
        ' Send a newsletter using Tautulli.\n\n            ```\n            Required parameters:\n                newsletter_id (int):      The ID number of the newsletter\n\n            Optional parameters:\n                None\n\n            Returns:\n                None\n            ```\n        '
        cherrypy.response.headers['Cache-Control'] = 'max-age=0,no-cache,no-store'
        test = 'test ' if notify_action == 'test' else ''
        if newsletter_id:
            newsletter = newsletters.get_newsletter_config(newsletter_id=newsletter_id)
            if newsletter:
                logger.debug('Sending %s%s newsletter.' % (test, newsletter['agent_label']))
                newsletter_handler.add_newsletter_each(newsletter_id=newsletter_id, notify_action=notify_action, subject=subject, body=body, message=message, **kwargs)
                return {'result': 'success', 'message': 'Newsletter queued.'}
            else:
                logger.debug('Unable to send %snewsletter, invalid newsletter_id %s.' % (test, newsletter_id))
                return {'result': 'error', 'message': 'Invalid newsletter id %s.' % newsletter_id}
        else:
            logger.debug('Unable to send %snotification, no newsletter_id received.' % test)
            return {'result': 'error', 'message': 'No newsletter id received.'}

    @cherrypy.expose
    def newsletter(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        request_uri = cherrypy.request.wsgi_environ['REQUEST_URI']
        if plexpy.CONFIG.NEWSLETTER_AUTH == 2:
            redirect_uri = request_uri.replace('/newsletter', '/newsletter_auth')
            raise cherrypy.HTTPRedirect(redirect_uri)
        elif plexpy.CONFIG.NEWSLETTER_AUTH == 1 and plexpy.CONFIG.NEWSLETTER_PASSWORD:
            if len(args) >= 2 and args[0] == 'image':
                return self.newsletter_auth(*args, **kwargs)
            elif kwargs.pop('key', None) == plexpy.CONFIG.NEWSLETTER_PASSWORD:
                return self.newsletter_auth(*args, **kwargs)
            else:
                return serve_template(template_name='newsletter_auth.html', title='Newsletter Login', uri=request_uri)
        else:
            return self.newsletter_auth(*args, **kwargs)

    @cherrypy.expose
    @requireAuth()
    def newsletter_auth(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if args:
            if len(args) >= 2 and args[0] == 'image':
                if args[1] == 'images':
                    resource_dir = os.path.join(str(plexpy.PROG_DIR), 'data/interfaces/default/')
                    try:
                        return serve_file(path=os.path.join(resource_dir, *args[1:]), content_type='image/png')
                    except NotFound:
                        return
                return self.image(args[1])
            if len(args) >= 2 and args[0] == 'id':
                newsletter_id_name = args[1]
                newsletter_uuid = None
            else:
                newsletter_id_name = None
                newsletter_uuid = args[0]
            newsletter = newsletter_handler.get_newsletter(newsletter_uuid=newsletter_uuid, newsletter_id_name=newsletter_id_name)
            return newsletter

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def newsletter_preview(self, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs['preview'] = 'true'
        return serve_template(template_name='newsletter_preview.html', title='Newsletter', kwargs=kwargs)

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def real_newsletter(self, newsletter_id=None, start_date=None, end_date=None, preview=False, raw=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if newsletter_id and newsletter_id != 'None':
            newsletter = newsletters.get_newsletter_config(newsletter_id=newsletter_id)
            if newsletter:
                newsletter_agent = newsletters.get_agent_class(newsletter_id=newsletter_id, newsletter_id_name=newsletter['id_name'], agent_id=newsletter['agent_id'], config=newsletter['config'], start_date=start_date, end_date=end_date, subject=newsletter['subject'], body=newsletter['body'], message=newsletter['message'])
                preview = helpers.bool_true(preview)
                raw = helpers.bool_true(raw)
                if raw:
                    cherrypy.response.headers['Content-Type'] = 'application/json;charset=UTF-8'
                    return json.dumps(newsletter_agent.raw_data(preview=preview)).encode('utf-8')
                return newsletter_agent.generate_newsletter(preview=preview)
            logger.error('Failed to retrieve newsletter: Invalid newsletter_id %s' % newsletter_id)
            return 'Failed to retrieve newsletter: invalid newsletter_id parameter'
        logger.error('Failed to retrieve newsletter: Missing newsletter_id parameter.')
        return 'Failed to retrieve newsletter: missing newsletter_id parameter'

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def support(self, **kwargs):
        if False:
            while True:
                i = 10
        return serve_template(template_name='support.html', title='Support')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @addtoapi()
    def status(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Get the current status of Tautulli.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                check (str):        database\n\n            Returns:\n                json:\n                    {"result": "success",\n                     "message": "Ok",\n                     }\n            ```\n        '
        cherrypy.response.headers['Cache-Control'] = 'max-age=0,no-cache,no-store'
        status = {'result': 'success', 'message': 'Ok'}
        if args or kwargs:
            if not cherrypy.request.path_info == '/api/v2' and plexpy.AUTH_ENABLED:
                cherrypy.request.config['auth.require'] = []
                check_auth()
            if 'database' in (args[:1] or kwargs.get('check')):
                result = database.integrity_check()
                status.update(result)
                if result['integrity_check'] == 'ok':
                    status['message'] = 'Database ok'
                else:
                    status['result'] = 'error'
                    status['message'] = 'Database not ok'
        return status

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @addtoapi()
    def server_status(self, *args, **kwargs):
        if False:
            return 10
        ' Get the current status of Tautulli\'s connection to the Plex server.\n\n            ```\n            Required parameters:\n                None\n\n            Optional parameters:\n                None\n\n            Returns:\n                json:\n                    {"result": "success",\n                     "connected": true,\n                     }\n            ```\n        '
        cherrypy.response.headers['Cache-Control'] = 'max-age=0,no-cache,no-store'
        status = {'result': 'success', 'connected': plexpy.PLEX_SERVER_UP}
        return status

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi('get_exports_table')
    def get_export_list(self, section_id=None, user_id=None, rating_key=None, **kwargs):
        if False:
            return 10
        ' Get the data on the Tautulli export tables.\n\n            ```\n            Required parameters:\n                section_id (str):               The id of the Plex library section, OR\n                user_id (str):                  The id of the Plex user, OR\n                rating_key (str):               The rating key of the exported item\n\n            Optional parameters:\n                order_column (str):             "added_at", "sort_title", "container", "bitrate", "video_codec",\n                                                "video_resolution", "video_framerate", "audio_codec", "audio_channels",\n                                                "file_size", "last_played", "play_count"\n                order_dir (str):                "desc" or "asc"\n                start (int):                    Row to start from, 0\n                length (int):                   Number of items to return, 25\n                search (str):                   A string to search for, "Thrones"\n\n            Returns:\n                json:\n                    {"draw": 1,\n                     "recordsTotal": 10,\n                     "recordsFiltered": 3,\n                     "data":\n                        [{"timestamp": 1602823644,\n                          "art_level": 0,\n                          "complete": 1,\n                          "custom_fields": "",\n                          "exists": true,\n                          "export_id": 42,\n                          "exported_items": 28,\n                          "file_format": "json",\n                          "file_size": 57793562,\n                          "filename": null,\n                          "individual_files": 1,\n                          "media_info_level": 1,\n                          "media_type": "collection",\n                          "media_type_title": "Collection",\n                          "metadata_level": 1,\n                          "rating_key": null,\n                          "section_id": 1,\n                          "thumb_level": 2,\n                          "title": "Library - Movies - Collection [1]",\n                          "total_items": 28,\n                          "user_id": null\n                          },\n                         {...},\n                         {...}\n                         ]\n                     }\n            ```\n        '
        if not kwargs.get('json_data'):
            dt_columns = [('timestamp', True, False), ('media_type_title', True, True), ('rating_key', True, True), ('title', True, True), ('file_format', True, True), ('metadata_level', True, True), ('media_info_level', True, True), ('custom_fields', True, True), ('file_size', True, False), ('complete', True, False)]
            kwargs['json_data'] = build_datatables_json(kwargs, dt_columns, 'timestamp')
        result = exporter.get_export_datatable(section_id=section_id, user_id=user_id, rating_key=rating_key, kwargs=kwargs)
        return result

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def export_metadata_modal(self, section_id=None, user_id=None, rating_key=None, media_type=None, sub_media_type=None, export_type=None, **kwargs):
        if False:
            print('Hello World!')
        file_formats = exporter.Export.FILE_FORMATS
        if media_type == 'photo_album':
            media_type = 'photoalbum'
        return serve_template(template_name='export_modal.html', title='Export Metadata', section_id=section_id, user_id=user_id, rating_key=rating_key, media_type=media_type, sub_media_type=sub_media_type, export_type=export_type, file_formats=file_formats)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def get_export_fields(self, media_type=None, sub_media_type=None, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Get a list of available custom export fields.\n\n            ```\n            Required parameters:\n                media_type (str):          The media type of the fields to return\n\n            Optional parameters:\n                sub_media_type (str):      The child media type for\n                                           collections (movie, show, artist, album, photoalbum),\n                                           or playlists (video, audio, photo)\n\n            Returns:\n                json:\n                    {"metadata_fields":\n                        [{"field": "addedAt", "level": 1},\n                         ...\n                         ],\n                     "media_info_fields":\n                        [{"field": "media.aspectRatio", "level": 1},\n                         ...\n                         ]\n                    }\n            ```\n        '
        custom_fields = exporter.get_custom_fields(media_type=media_type, sub_media_type=sub_media_type)
        return custom_fields

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def export_metadata(self, section_id=None, user_id=None, rating_key=None, file_format='csv', metadata_level=1, media_info_level=1, thumb_level=0, art_level=0, custom_fields='', export_type='all', individual_files=False, **kwargs):
        if False:
            return 10
        ' Export library or media metadata to a file\n\n            ```\n            Required parameters:\n                section_id (int):          The section id of the library items to export, OR\n                user_id (int):             The user id of the playlist items to export, OR\n                rating_key (int):          The rating key of the media item to export\n\n            Optional parameters:\n                file_format (str):         csv (default), json, xml, or m3u\n                metadata_level (int):      The level of metadata to export (default 1)\n                media_info_level (int):    The level of media info to export (default 1)\n                thumb_level (int):         The level of poster/cover images to export (default 0)\n                art_level (int):           The level of background artwork images to export (default 0)\n                custom_fields (str):       Comma separated list of custom fields to export\n                                           in addition to the export level selected\n                export_type (str):         \'collection\' or \'playlist\' for library/user export,\n                                           otherwise default to all library items\n                individual_files (bool):   Export each item as an individual file for library/user export.\n\n            Returns:\n                json:\n                    {"export_id": 1}\n            ```\n        '
        individual_files = helpers.bool_true(individual_files)
        result = exporter.Export(section_id=section_id, user_id=user_id, rating_key=rating_key, file_format=file_format, metadata_level=metadata_level, media_info_level=media_info_level, thumb_level=thumb_level, art_level=art_level, custom_fields=custom_fields, export_type=export_type, individual_files=individual_files).export()
        if isinstance(result, int):
            return {'result': 'success', 'message': 'Metadata export has started.', 'export_id': result}
        else:
            return {'result': 'error', 'message': result}

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def view_export(self, export_id=None, **kwargs):
        if False:
            return 10
        ' Download an exported metadata file\n\n            ```\n            Required parameters:\n                export_id (int):          The row id of the exported file to view\n\n            Optional parameters:\n                None\n\n            Returns:\n                download\n            ```\n        '
        result = exporter.get_export(export_id=export_id)
        if result and result['complete'] == 1 and result['exists'] and (not result['individual_files']):
            filepath = exporter.get_export_filepath(result['title'], result['timestamp'], result['filename'])
            if result['file_format'] == 'csv':
                with open(filepath, 'r', encoding='utf-8') as infile:
                    reader = csv.DictReader(infile)
                    table = '<table><tr><th>' + '</th><th>'.join(reader.fieldnames) + '</th></tr><tr>' + '</tr><tr>'.join(('<td>' + '</td><td>'.join(row.values()) + '</td>' for row in reader)) + '</tr></table>'
                    style = '<style>body {margin: 0;}table {border-collapse: collapse; overflow-y: auto; height: 100px;} th {position: sticky; top: 0; background: #ddd; box-shadow: inset 1px 1px #000, 0 1px #000;}td {box-shadow: inset 1px -1px #000;}th, td {padding: 3px; white-space: nowrap;}</style>'
                return '{style}<pre>{table}</pre>'.format(style=style, table=table)
            elif result['file_format'] == 'json':
                return serve_file(filepath, name=result['filename'], content_type='application/json;charset=UTF-8')
            elif result['file_format'] == 'xml':
                return serve_file(filepath, name=result['filename'], content_type='application/xml;charset=UTF-8')
            elif result['file_format'] == 'm3u':
                return serve_file(filepath, name=result['filename'], content_type='text/plain;charset=UTF-8')
        else:
            if result and result.get('complete') == 0:
                msg = 'Export is still being processed.'
            elif result and result.get('complete') == -1:
                msg = 'Export failed to process.'
            elif result and (not result.get('exists')):
                msg = 'Export file does not exist.'
            else:
                msg = 'Invalid export_id provided.'
            cherrypy.response.headers['Content-Type'] = 'application/json;charset=UTF-8'
            return json.dumps({'result': 'error', 'message': msg}).encode('utf-8')

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    @addtoapi()
    def download_export(self, export_id=None, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Download an exported metadata file\n\n            ```\n            Required parameters:\n                export_id (int):          The row id of the exported file to download\n\n            Optional parameters:\n                None\n\n            Returns:\n                download\n            ```\n        '
        result = exporter.get_export(export_id=export_id)
        if result and result['complete'] == 1 and result['exists']:
            if result['thumb_level'] or result['art_level'] or result['individual_files']:
                directory = exporter.format_export_directory(result['title'], result['timestamp'])
                dirpath = exporter.get_export_dirpath(directory)
                zip_filename = '{}.zip'.format(directory)
                buffer = BytesIO()
                temp_zip = zipfile.ZipFile(buffer, 'w')
                helpers.zipdir(dirpath, temp_zip)
                temp_zip.close()
                return serve_fileobj(buffer.getvalue(), content_type='application/zip', disposition='attachment', name=zip_filename)
            else:
                filepath = exporter.get_export_filepath(result['title'], result['timestamp'], result['filename'])
                return serve_download(filepath, name=result['filename'])
        else:
            if result and result.get('complete') == 0:
                msg = 'Export is still being processed.'
            elif result and result.get('complete') == -1:
                msg = 'Export failed to process.'
            elif result and (not result.get('exists')):
                msg = 'Export file does not exist.'
            else:
                msg = 'Invalid export_id provided.'
            cherrypy.response.headers['Content-Type'] = 'application/json;charset=UTF-8'
            return json.dumps({'result': 'error', 'message': msg}).encode('utf-8')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @requireAuth(member_of('admin'))
    @addtoapi()
    def delete_export(self, export_id=None, delete_all=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        " Delete exports from Tautulli.\n\n            ```\n            Required parameters:\n                export_id (int):          The row id of the exported file to delete\n\n            Optional parameters:\n                delete_all (bool):        'true' to delete all exported files\n\n            Returns:\n                None\n            ```\n        "
        if helpers.bool_true(delete_all):
            result = exporter.delete_all_exports()
            if result:
                return {'result': 'success', 'message': 'All exports deleted successfully.'}
            else:
                return {'result': 'error', 'message': 'Failed to delete all exports.'}
        else:
            result = exporter.delete_export(export_id=export_id)
            if result:
                return {'result': 'success', 'message': 'Export deleted successfully.'}
            else:
                return {'result': 'error', 'message': 'Failed to delete export.'}

    @cherrypy.expose
    @requireAuth(member_of('admin'))
    def exporter_docs(self, **kwargs):
        if False:
            while True:
                i = 10
        return '<pre>' + exporter.build_export_docs() + '</pre>'