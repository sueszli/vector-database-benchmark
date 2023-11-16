"""Hook for asynchronous checkpointing.

This hook dispatches checkpoint writing operations in a separate thread to
allow execution to continue on the main thread.
"""
import os
import threading
import time
from typing import Any, List, Optional, Text
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.summary_io import SummaryWriterCache
_END_TIME_OF_LAST_WRITE = None
_END_TIME_OF_LAST_WRITE_LOCK = threading.Lock()
_ASYNC_CHECKPOINT_V1 = 'async_checkpoint_v1'

def _get_duration_microseconds(start_time_seconds, end_time_seconds) -> int:
    if False:
        print('Hello World!')
    'Returns the duration between start and end time in microseconds.'
    return max(int((end_time_seconds - start_time_seconds) * 1000000), 0)

class AsyncCheckpointSaverHook(basic_session_run_hooks.CheckpointSaverHook):
    """Saves checkpoints every N steps or seconds."""

    def __init__(self, checkpoint_dir: Text, save_secs: Optional[int]=None, save_steps: Optional[int]=None, saver: Optional[saver_lib.Saver]=None, checkpoint_basename: Text='model.ckpt', scaffold: Optional[monitored_session.Scaffold]=None, listeners: Optional[List[basic_session_run_hooks.CheckpointSaverListener]]=None):
        if False:
            return 10
        'Initializes a `CheckpointSaverHook`.\n\n    Args:\n      checkpoint_dir: `str`, base directory for the checkpoint files.\n      save_secs: `int`, save every N secs.\n      save_steps: `int`, save every N steps.\n      saver: `Saver` object, used for saving.\n      checkpoint_basename: `str`, base name for the checkpoint files.\n      scaffold: `Scaffold`, use to get saver object.\n      listeners: List of `CheckpointSaverListener` subclass instances. Used for\n        callbacks that run immediately before or after this hook saves the\n        checkpoint.\n\n    Raises:\n      ValueError: One of `save_steps` or `save_secs` should be set.\n      ValueError: At most one of `saver` or `scaffold` should be set.\n    '
        save_path = os.path.join(checkpoint_dir, checkpoint_basename)
        logging.info('Create AsyncCheckpointSaverHook saving to path\n%s', save_path)
        if listeners:
            logging.info(' with %d listener(s).', len(listeners))
        if saver is not None and scaffold is not None:
            raise ValueError('You cannot provide both saver and scaffold.')
        self._saver = saver
        self._save_thread = None
        self._write_graph_thread = None
        self._checkpoint_dir = checkpoint_dir
        self._save_path = save_path
        self._scaffold = scaffold
        self._timer = basic_session_run_hooks.SecondOrStepTimer(every_secs=save_secs, every_steps=save_steps)
        self._listeners = listeners or []
        self._steps_per_run = 1
        self._summary_writer = None
        self._global_step_tensor = None
        self._last_checkpoint_step = None
        global _END_TIME_OF_LAST_WRITE
        with _END_TIME_OF_LAST_WRITE_LOCK:
            if _END_TIME_OF_LAST_WRITE is None:
                _END_TIME_OF_LAST_WRITE = time.time()

    def _set_steps_per_run(self, steps_per_run):
        if False:
            for i in range(10):
                print('nop')
        self._steps_per_run = steps_per_run

    def begin(self):
        if False:
            for i in range(10):
                print('nop')
        self._summary_writer = SummaryWriterCache.get(self._checkpoint_dir)
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        if self._global_step_tensor is None:
            raise RuntimeError('Global step should be created to use CheckpointSaverHook.')
        for l in self._listeners:
            l.begin()

    def after_create_session(self, session: session_lib.Session, coord: Any):
        if False:
            print('Hello World!')
        global_step = session.run(self._global_step_tensor)

        def _write_graph_fn(self):
            if False:
                for i in range(10):
                    print('nop')
            training_util.write_graph(ops.get_default_graph().as_graph_def(add_shapes=True), self._checkpoint_dir, 'graph.pbtxt')
        self._write_graph_thread = threading.Thread(target=_write_graph_fn, args=[self])
        self._write_graph_thread.start()
        saver_def = self._get_saver().saver_def if self._get_saver() else None
        graph = ops.get_default_graph()
        meta_graph_def = meta_graph.create_meta_graph_def(graph_def=graph.as_graph_def(add_shapes=True), saver_def=saver_def)
        if self._summary_writer is None:
            raise ValueError('Summary writer is not initialised')
        self._summary_writer.add_graph(graph)
        self._summary_writer.add_meta_graph(meta_graph_def)
        self._save(session, global_step)
        self._timer.update_last_triggered_step(global_step)

    def before_run(self, run_context: Any):
        if False:
            print('Hello World!')
        return session_run_hook.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context: session_run_hook.SessionRunContext, run_values: Any):
        if False:
            while True:
                i = 10
        global_step = run_context.session.run(self._global_step_tensor)
        if self._timer.should_trigger_for_step(global_step):
            self._timer.update_last_triggered_step(global_step)
            logging.info('Triggering checkpoint. %s', global_step)
            if self._save(run_context.session, global_step):
                run_context.request_stop()

    def end(self, session: session_lib.Session):
        if False:
            return 10
        if self._save_thread:
            logging.info('Waiting for any pending checkpoints to finish.')
            self._save_thread.join()
        if self._write_graph_thread:
            logging.info('Waiting for any pending write_graph to finish.')
            self._write_graph_thread.join()
        last_step = session.run(self._global_step_tensor)
        if self._last_checkpoint_step != last_step:
            self._save(session, last_step, asynchronous=False)
        for l in self._listeners:
            l.end(session, last_step)

    def _save(self, session, step, asynchronous=True):
        if False:
            for i in range(10):
                print('nop')
        'Saves the latest checkpoint, returns should_stop.'

        def _save_fn():
            if False:
                print('Hello World!')
            'Run the saver process.'
            logging.info('Saving checkpoints for %d into %s.', step, self._save_path)
            start_time = time.time()
            for l in self._listeners:
                l.before_save(session, step)
            self._get_saver().save(session, self._save_path, global_step=step)
            if self._summary_writer is None:
                raise ValueError('Summary writer is not initialised')
            self._summary_writer.add_session_log(event_pb2.SessionLog(status=event_pb2.SessionLog.CHECKPOINT, checkpoint_path=self._save_path), step)
            for l in self._listeners:
                l.after_save(session, step)
            end_time = time.time()
            metrics.AddAsyncCheckpointWriteDuration(api_label=_ASYNC_CHECKPOINT_V1, microseconds=_get_duration_microseconds(start_time, end_time))
            global _END_TIME_OF_LAST_WRITE
            with _END_TIME_OF_LAST_WRITE_LOCK:
                metrics.AddTrainingTimeSaved(api_label=_ASYNC_CHECKPOINT_V1, microseconds=_get_duration_microseconds(_END_TIME_OF_LAST_WRITE, start_time))
            _END_TIME_OF_LAST_WRITE = start_time
            logging.info('Checkpoint actual writing time: (%.3f sec)', end_time - start_time)
            logging.info('Checkpoint finished for %d into %s.', step, self._save_path)
        blocking_start_time = time.time()

        def end_of_blocking_time():
            if False:
                print('Hello World!')
            blocking_end_time = time.time()
            metrics.AddCheckpointWriteDuration(api_label=_ASYNC_CHECKPOINT_V1, microseconds=_get_duration_microseconds(blocking_start_time, blocking_end_time))
        if not asynchronous:
            self._last_checkpoint_step = step
            _save_fn()
            end_of_blocking_time()
            return
        if self._save_thread is not None:
            self._save_thread.join(timeout=0.1)
            if self._save_thread.is_alive():
                logging.info('Saver thread still in progress, skipping checkpoint.')
                end_of_blocking_time()
                return
        self._last_checkpoint_step = step
        self._save_thread = threading.Thread(target=_save_fn)
        self._save_thread.start()
        end_of_blocking_time()

    def _get_saver(self):
        if False:
            i = 10
            return i + 15
        if self._saver is not None:
            return self._saver
        elif self._scaffold is not None:
            return self._scaffold.saver
        collection_key = ops.GraphKeys.SAVERS
        savers = ops.get_collection(collection_key)
        if not savers:
            raise RuntimeError('No items in collection {}. Please add a saver to the collection or provide a saver or scaffold.'.format(collection_key))
        elif len(savers) > 1:
            raise RuntimeError('More than one item in collection {}. Please indicate which one to use by passing it to the constructor.'.format(collection_key))
        self._saver = savers[0]
        return savers[0]