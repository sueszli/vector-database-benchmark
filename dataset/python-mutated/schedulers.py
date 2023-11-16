import logging
from celery.utils.log import get_logger
from django.db import close_old_connections
from django.core.exceptions import ObjectDoesNotExist
from django.db.utils import DatabaseError, InterfaceError
from django_celery_beat.schedulers import DatabaseScheduler as DJDatabaseScheduler
logger = get_logger(__name__)
(debug, info, warning) = (logger.debug, logger.info, logger.warning)
__all__ = ['DatabaseScheduler']

class DatabaseScheduler(DJDatabaseScheduler):

    def sync(self):
        if False:
            i = 10
            return i + 15
        if logger.isEnabledFor(logging.DEBUG):
            debug('Writing entries...')
        _tried = set()
        _failed = set()
        try:
            close_old_connections()
            while self._dirty:
                name = self._dirty.pop()
                try:
                    '\n                    ::Debug Description (2023.07.10)::\n                    \n                    如果调用 self.schedule 可能会导致 self.save() 方法之前重新获取数据库中的数据, 而不是临时设置的 last_run_at 数据\n                    \n                    如果这里调用 self.schedule\n                    那么可能会导致调用 save 的 self.schedule[name] 的 last_run_at 是从数据库中获取回来的老数据\n                    而不是任务执行后临时设置的 last_run_at (在 __next__() 方法中设置的)\n                    当 `max_interval` 间隔之后, 下一个任务检测周期还是会再次执行任务\n                    \n                    ::Demo::\n                    任务信息:\n                        beat config: max_interval = 60s\n                        \n                        任务名称: cap\n                        任务执行周期: 每 3 分钟执行一次\n                        任务最后执行时间: 18:00\n                        \n                    任务第一次执行: 18:03 (执行时设置 last_run_at = 18:03, 此时在内存中)\n                    \n                    任务执行完成后, \n                    检测到需要 sync, sync 中调用了 self.schedule,\n                    self.schedule 中发现 schedule_changed() 为 True, 需要调用 all_as_schedule()\n                    此时，sync 中调用的 self.schedule[name] 的 last_run_at 是 18:00\n                    这时候在 self.sync() 进行 self.save()\n                    \n                    \n                    beat: Waking up 60s ...\n                    \n                    任务第二次执行: 18:04 (因为获取回来的 last_run_at 是 18:00, entry.is_due() == True)\n                    \n                    ::解决方法::\n                    所以这里为了避免从数据库中获取，直接使用 _schedule #\n                    '
                    self._schedule[name].save()
                    _tried.add(name)
                except (KeyError, TypeError, ObjectDoesNotExist):
                    _failed.add(name)
        except DatabaseError as exc:
            logger.exception('Database error while sync: %r', exc)
        except InterfaceError:
            warning('DatabaseScheduler: InterfaceError in sync(), waiting to retry in next call...')
        finally:
            self._dirty |= _failed