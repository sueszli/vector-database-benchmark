"""Django-specific customization."""
import os
import sys
import warnings
from datetime import datetime
from importlib import import_module
from typing import IO, TYPE_CHECKING, Any, List, Optional, cast
from kombu.utils.imports import symbol_by_name
from kombu.utils.objects import cached_property
from celery import _state, signals
from celery.exceptions import FixupWarning, ImproperlyConfigured
if TYPE_CHECKING:
    from types import ModuleType
    from typing import Protocol
    from django.db.utils import ConnectionHandler
    from celery.app.base import Celery
    from celery.app.task import Task

    class DjangoDBModule(Protocol):
        connections: ConnectionHandler
__all__ = ('DjangoFixup', 'fixup')
ERR_NOT_INSTALLED = "Environment variable DJANGO_SETTINGS_MODULE is defined\nbut Django isn't installed.  Won't apply Django fix-ups!\n"

def _maybe_close_fd(fh: IO) -> None:
    if False:
        while True:
            i = 10
    try:
        os.close(fh.fileno())
    except (AttributeError, OSError, TypeError):
        pass

def _verify_django_version(django: 'ModuleType') -> None:
    if False:
        i = 10
        return i + 15
    if django.VERSION < (1, 11):
        raise ImproperlyConfigured('Celery 5.x requires Django 1.11 or later.')

def fixup(app: 'Celery', env: str='DJANGO_SETTINGS_MODULE') -> Optional['DjangoFixup']:
    if False:
        while True:
            i = 10
    'Install Django fixup if settings module environment is set.'
    SETTINGS_MODULE = os.environ.get(env)
    if SETTINGS_MODULE and 'django' not in app.loader_cls.lower():
        try:
            import django
        except ImportError:
            warnings.warn(FixupWarning(ERR_NOT_INSTALLED))
        else:
            _verify_django_version(django)
            return DjangoFixup(app).install()
    return None

class DjangoFixup:
    """Fixup installed when using Django."""

    def __init__(self, app: 'Celery'):
        if False:
            print('Hello World!')
        self.app = app
        if _state.default_app is None:
            self.app.set_default()
        self._worker_fixup: Optional['DjangoWorkerFixup'] = None

    def install(self) -> 'DjangoFixup':
        if False:
            return 10
        sys.path.insert(0, os.getcwd())
        self._settings = symbol_by_name('django.conf:settings')
        self.app.loader.now = self.now
        signals.import_modules.connect(self.on_import_modules)
        signals.worker_init.connect(self.on_worker_init)
        return self

    @property
    def worker_fixup(self) -> 'DjangoWorkerFixup':
        if False:
            i = 10
            return i + 15
        if self._worker_fixup is None:
            self._worker_fixup = DjangoWorkerFixup(self.app)
        return self._worker_fixup

    @worker_fixup.setter
    def worker_fixup(self, value: 'DjangoWorkerFixup') -> None:
        if False:
            print('Hello World!')
        self._worker_fixup = value

    def on_import_modules(self, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        self.worker_fixup.validate_models()

    def on_worker_init(self, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.worker_fixup.install()

    def now(self, utc: bool=False) -> datetime:
        if False:
            print('Hello World!')
        return datetime.utcnow() if utc else self._now()

    def autodiscover_tasks(self) -> List[str]:
        if False:
            return 10
        from django.apps import apps
        return [config.name for config in apps.get_app_configs()]

    @cached_property
    def _now(self) -> datetime:
        if False:
            print('Hello World!')
        return symbol_by_name('django.utils.timezone:now')

class DjangoWorkerFixup:
    _db_recycles = 0

    def __init__(self, app: 'Celery') -> None:
        if False:
            i = 10
            return i + 15
        self.app = app
        self.db_reuse_max = self.app.conf.get('CELERY_DB_REUSE_MAX', None)
        self._db = cast('DjangoDBModule', import_module('django.db'))
        self._cache = import_module('django.core.cache')
        self._settings = symbol_by_name('django.conf:settings')
        self.interface_errors = (symbol_by_name('django.db.utils.InterfaceError'),)
        self.DatabaseError = symbol_by_name('django.db:DatabaseError')

    def django_setup(self) -> None:
        if False:
            return 10
        import django
        django.setup()

    def validate_models(self) -> None:
        if False:
            while True:
                i = 10
        from django.core.checks import run_checks
        self.django_setup()
        if not os.environ.get('CELERY_SKIP_CHECKS'):
            run_checks()

    def install(self) -> 'DjangoWorkerFixup':
        if False:
            i = 10
            return i + 15
        signals.beat_embedded_init.connect(self.close_database)
        signals.task_prerun.connect(self.on_task_prerun)
        signals.task_postrun.connect(self.on_task_postrun)
        signals.worker_process_init.connect(self.on_worker_process_init)
        self.close_database()
        self.close_cache()
        return self

    def on_worker_process_init(self, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        if os.environ.get('FORKED_BY_MULTIPROCESSING'):
            self.validate_models()
        for c in self._db.connections.all():
            if c and c.connection:
                self._maybe_close_db_fd(c.connection)
        self._close_database(force=True)
        self.close_cache()

    def _maybe_close_db_fd(self, fd: IO) -> None:
        if False:
            i = 10
            return i + 15
        try:
            _maybe_close_fd(fd)
        except self.interface_errors:
            pass

    def on_task_prerun(self, sender: 'Task', **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        'Called before every task.'
        if not getattr(sender.request, 'is_eager', False):
            self.close_database()

    def on_task_postrun(self, sender: 'Task', **kwargs: Any) -> None:
        if False:
            return 10
        if not getattr(sender.request, 'is_eager', False):
            self.close_database()
            self.close_cache()

    def close_database(self, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        if not self.db_reuse_max:
            return self._close_database()
        if self._db_recycles >= self.db_reuse_max * 2:
            self._db_recycles = 0
            self._close_database()
        self._db_recycles += 1

    def _close_database(self, force: bool=False) -> None:
        if False:
            print('Hello World!')
        for conn in self._db.connections.all():
            try:
                if force:
                    conn.close()
                else:
                    conn.close_if_unusable_or_obsolete()
            except self.interface_errors:
                pass
            except self.DatabaseError as exc:
                str_exc = str(exc)
                if 'closed' not in str_exc and 'not connected' not in str_exc:
                    raise

    def close_cache(self) -> None:
        if False:
            return 10
        try:
            self._cache.close_caches()
        except (TypeError, AttributeError):
            pass