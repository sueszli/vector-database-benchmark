"""Implementation of the SessionRunHook for preemptible Cloud TPUs."""
import logging as _logging
import os
import threading
import time
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook

class CloudTPUPreemptedHook(session_run_hook.SessionRunHook):
    """The SessionRunHook for preemptible Cloud TPUs.

  This is an implementation of SessionRunHook for the pre-emptible Google Cloud
  TPU service. It attempts to close the session if the TPU is preempted, and
  exits the coordinator process if the session cannot be closed.
  """

    def __init__(self, cluster):
        if False:
            while True:
                i = 10
        self._cluster = cluster

    def after_create_session(self, session, coord):
        if False:
            return 10
        if tpu_cluster_resolver.is_running_in_gce():
            self._tpu_poller = _TPUPollingThread(self._cluster, session)
            self._tpu_poller.start()

    def end(self, session):
        if False:
            return 10
        self._tpu_poller.stop()

class _TPUPollingThread(threading.Thread):
    """A thread that polls the state of a TPU node.

  When the node transitions into a TERMINAL state (PREEMPTED, TERMINATED)
  that's considered as not recoverable by the underlying infrastructure,
  it attempts to close the session, and exits the entire process if the
  session.close() stucks.
  """

    def __init__(self, cluster, session):
        if False:
            i = 10
            return i + 15
        super(_TPUPollingThread, self).__init__()
        self.daemon = True
        self._running = True
        self._session_closed = False
        self._cluster = cluster
        self._session = session
        self._interval = 30
        for name in ['googleapiclient.discovery', 'oauth2client.client']:
            _logging.getLogger(name).setLevel(_logging.WARNING)

    def stop(self):
        if False:
            print('Hello World!')
        self._running = False
        self._session_closed = True
        self.join()

    def run(self):
        if False:
            while True:
                i = 10
        if not tpu_cluster_resolver.is_running_in_gce():
            logging.warning('TPUPollingThread is running in a non-GCE environment, exiting...')
            self._running = False
            return
        while self._running:
            recoverable = self._cluster._cloud_tpu_client.recoverable()
            if not recoverable:
                logging.warning('TPUPollingThread found TPU %s in state %s', self._cluster._tpu, self._cluster._cloud_tpu_client.state())
                os._exit(1)
            time.sleep(self._interval)