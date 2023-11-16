import logging
from unittest.mock import patch
from twisted.internet.defer import inlineCallbacks
from golemapp import main
from golem.client import Client
from golem.task.taskserver import TaskServer

def on_exception():
    if False:
        while True:
            i = 10
    logging.critical('#### Integration test failed ####')
client_change_config_orig = Client.change_config

def client_change_config(self: Client, *args, **kwargs):
    if False:
        while True:
            i = 10
    try:
        client_change_config_orig(self, *args, **kwargs)
    except:
        on_exception()
task_server_change_config_orig = TaskServer.change_config

@inlineCallbacks
def task_server_change_config(self: TaskServer, *args, **kwargs):
    if False:
        while True:
            i = 10
    try:
        yield task_server_change_config_orig(self, *args, **kwargs)
    except:
        on_exception()
with patch('golem.client.Client.change_config', client_change_config), patch('golem.task.taskserver.TaskServer.change_config', task_server_change_config):
    main()