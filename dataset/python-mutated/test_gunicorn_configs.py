import os
import random
from six.moves import http_client
import unittest2
import requests
import eventlet
from eventlet.green import subprocess
from st2common.models.utils import profiling
from st2common.util.shell import kill_process
from st2tests.base import IntegrationTestCase
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ST2_CONFIG_PATH = os.path.join(BASE_DIR, '../../../conf/st2.tests.conf')

class GunicornWSGIEntryPointTestCase(IntegrationTestCase):

    @unittest2.skipIf(profiling.is_enabled(), 'Profiling is enabled')
    def test_st2api_wsgi_entry_point(self):
        if False:
            print('Hello World!')
        port = random.randint(10000, 30000)
        cmd = 'gunicorn st2api.wsgi:application -k eventlet -b "127.0.0.1:%s" --workers 1' % port
        env = os.environ.copy()
        env['ST2_CONFIG_PATH'] = ST2_CONFIG_PATH
        process = subprocess.Popen(cmd, env=env, shell=True, preexec_fn=os.setsid)
        try:
            self.add_process(process=process)
            eventlet.sleep(8)
            self.assertProcessIsRunning(process=process)
            response = requests.get('http://127.0.0.1:%s/v1/actions' % port)
            self.assertEqual(response.status_code, http_client.OK)
        finally:
            kill_process(process)

    @unittest2.skipIf(profiling.is_enabled(), 'Profiling is enabled')
    def test_st2auth(self):
        if False:
            for i in range(10):
                print('nop')
        port = random.randint(10000, 30000)
        cmd = 'gunicorn st2auth.wsgi:application -k eventlet -b "127.0.0.1:%s" --workers 1' % port
        env = os.environ.copy()
        env['ST2_CONFIG_PATH'] = ST2_CONFIG_PATH
        process = subprocess.Popen(cmd, env=env, shell=True, preexec_fn=os.setsid)
        try:
            self.add_process(process=process)
            eventlet.sleep(8)
            self.assertProcessIsRunning(process=process)
            response = requests.post('http://127.0.0.1:%s/tokens' % port)
            self.assertEqual(response.status_code, http_client.UNAUTHORIZED)
        finally:
            kill_process(process)