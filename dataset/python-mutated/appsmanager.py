import os
from configparser import ConfigParser
from collections import OrderedDict
from importlib import import_module
from typing import Dict, List, TYPE_CHECKING, Tuple, Type, Optional
from golem.config.active import APP_MANAGER_CONFIG_FILES, CONCENT_SUPPORTED_APPS
from golem.core.common import get_golem_path
from golem.environments.environment import SupportStatus
from golem.task.taskbase import Task
if TYPE_CHECKING:
    from golem.environments.environment import Environment
    from golem.task.taskbase import TaskBuilder, TaskTypeInfo
    from apps.core.benchmark.benchmarkrunner import CoreBenchmark

class App(object):
    """ Basic Golem App Representation """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.env: Type['Environment'] = None
        self.builder: Type['TaskBuilder'] = None
        self.task_type_info: Type['TaskTypeInfo'] = None
        self.benchmark: Type['CoreBenchmark'] = None
        self.benchmark_builder: Type['TaskBuilder'] = None

    @property
    def concent_supported(self):
        if False:
            print('Hello World!')
        return self.task_type_info().id in CONCENT_SUPPORTED_APPS

class AppsManager(object):
    """ Temporary solution for apps detection and management. """

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.apps: Dict[str, App] = OrderedDict()
        self.task_types: Dict[str, App] = dict()

    def load_all_apps(self) -> None:
        if False:
            while True:
                i = 10
        for config_file in APP_MANAGER_CONFIG_FILES:
            self._load_apps(config_file)

    def _load_apps(self, apps_config_file) -> None:
        if False:
            i = 10
            return i + 15
        parser = ConfigParser()
        config_path = os.path.join(get_golem_path(), apps_config_file)
        with open(config_path) as config_file:
            parser.read_file(config_file)
        for section in parser.sections():
            app = App()
            for opt in vars(app):
                full_name = parser.get(section, opt)
                (package, name) = full_name.rsplit('.', 1)
                module = import_module(package)
                setattr(app, opt, getattr(module, name))
            self.apps[section] = app
            self.task_types[app.task_type_info().id] = app

    def get_env_list(self) -> List['Environment']:
        if False:
            while True:
                i = 10
        return [app.env() for app in self.apps.values()]

    def get_benchmarks(self) -> Dict[str, Tuple['CoreBenchmark', Type['TaskBuilder']]]:
        if False:
            return 10
        ' Returns list of data representing benchmark for registered app\n        :return dict: dictionary, where environment ids are the keys and values\n        are defined as pairs of instance of Benchmark and class of task builder\n        '
        benchmarks = dict()
        for app in self.apps.values():
            env = app.env()
            if not self._benchmark_enabled(env):
                continue
            benchmarks[env.get_id()] = (app.benchmark(), app.benchmark_builder)
        return benchmarks

    @staticmethod
    def _benchmark_enabled(env) -> bool:
        if False:
            print('Hello World!')
        return env.check_support() == SupportStatus.ok()

    def get_app(self, task_type_id: str) -> App:
        if False:
            i = 10
            return i + 15
        return self.task_types.get(task_type_id)

    def get_app_for_env(self, env_id: str) -> Optional[App]:
        if False:
            for i in range(10):
                print('nop')
        for app in self.apps.values():
            if app.env.get_id() == env_id:
                return app
        return None

    def get_task_class_for_env(self, env_id: str):
        if False:
            print('Hello World!')
        app = self.get_app_for_env(env_id)
        return app.builder.TASK_CLASS if app else Task