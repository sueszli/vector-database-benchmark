"""Dispatcher process which orchestrates distributed :class:`~gensim.models.lsimodel.LsiModel` computations.
Run this script only once, on any node in your cluster.

Notes
-----
The dispatcher expects to find worker scripts already running. Make sure you run as many workers as you like on
your machines **before** launching the dispatcher.


How to use distributed LSI
--------------------------

#. Install needed dependencies (Pyro4) ::

    pip install gensim[distributed]

#. Setup serialization (on each machine) ::

    export PYRO_SERIALIZERS_ACCEPTED=pickle
    export PYRO_SERIALIZER=pickle

#. Run nameserver ::

    python -m Pyro4.naming -n 0.0.0.0 &

#. Run workers (on each machine) ::

    python -m gensim.models.lsi_worker &

#. Run dispatcher ::

    python -m gensim.models.lsi_dispatcher &

#. Run :class:`~gensim.models.lsimodel.LsiModel` in distributed mode:

    .. sourcecode:: pycon

        >>> from gensim.test.utils import common_corpus, common_dictionary
        >>> from gensim.models import LsiModel
        >>>
        >>> model = LsiModel(common_corpus, id2word=common_dictionary, distributed=True)

Command line arguments
----------------------

.. program-output:: python -m gensim.models.lsi_dispatcher --help
   :ellipsis: 0, -5

"""
import os
import sys
import logging
import argparse
import threading
import time
from queue import Queue
import Pyro4
from gensim import utils
logger = logging.getLogger(__name__)
MAX_JOBS_QUEUE = 10
HUGE_TIMEOUT = 365 * 24 * 60 * 60

class Dispatcher:
    """Dispatcher object that communicates and coordinates individual workers.

    Warnings
    --------
    There should never be more than one dispatcher running at any one time.

    """

    def __init__(self, maxsize=0):
        if False:
            print('Hello World!')
        'Partly initialize the dispatcher.\n\n        A full initialization (including initialization of the workers) requires a call to\n        :meth:`~gensim.models.lsi_dispatcher.Dispatcher.initialize`\n\n        Parameters\n        ----------\n        maxsize : int, optional\n            Maximum number of jobs to be kept pre-fetched in the queue.\n\n        '
        self.maxsize = maxsize
        self.workers = {}
        self.callback = None

    @Pyro4.expose
    def initialize(self, **model_params):
        if False:
            while True:
                i = 10
        'Fully initialize the dispatcher and all its workers.\n\n        Parameters\n        ----------\n        **model_params\n            Keyword parameters used to initialize individual workers\n            (gets handed all the way down to :meth:`gensim.models.lsi_worker.Worker.initialize`).\n            See :class:`~gensim.models.lsimodel.LsiModel`.\n\n        Raises\n        ------\n        RuntimeError\n            When no workers are found (the :mod:`gensim.model.lsi_worker` script must be ran beforehand).\n\n        '
        self.jobs = Queue(maxsize=self.maxsize)
        self.lock_update = threading.Lock()
        self._jobsdone = 0
        self._jobsreceived = 0
        self.workers = {}
        with utils.getNS() as ns:
            self.callback = Pyro4.Proxy('PYRONAME:gensim.lsi_dispatcher')
            for (name, uri) in ns.list(prefix='gensim.lsi_worker').items():
                try:
                    worker = Pyro4.Proxy(uri)
                    workerid = len(self.workers)
                    logger.info('registering worker #%i from %s', workerid, uri)
                    worker.initialize(workerid, dispatcher=self.callback, **model_params)
                    self.workers[workerid] = worker
                except Pyro4.errors.PyroError:
                    logger.exception('unresponsive worker at %s, deleting it from the name server', uri)
                    ns.remove(name)
        if not self.workers:
            raise RuntimeError('no workers found; run some lsi_worker scripts on your machines first!')

    @Pyro4.expose
    def getworkers(self):
        if False:
            print('Hello World!')
        'Get pyro URIs of all registered workers.\n\n        Returns\n        -------\n        list of URIs\n            The pyro URIs for each worker.\n\n        '
        return [worker._pyroUri for worker in self.workers.values()]

    @Pyro4.expose
    def getjob(self, worker_id):
        if False:
            print('Hello World!')
        'Atomically pop a job from the queue.\n\n        Parameters\n        ----------\n        worker_id : int\n            The worker that requested the job.\n\n        Returns\n        -------\n        iterable of iterable of (int, float)\n            The corpus in BoW format.\n\n        '
        logger.info('worker #%i requesting a new job', worker_id)
        job = self.jobs.get(block=True, timeout=1)
        logger.info('worker #%i got a new job (%i left)', worker_id, self.jobs.qsize())
        return job

    @Pyro4.expose
    def putjob(self, job):
        if False:
            for i in range(10):
                print('nop')
        'Atomically add a job to the queue.\n\n        Parameters\n        ----------\n        job : iterable of list of (int, float)\n            The corpus in BoW format.\n\n        '
        self._jobsreceived += 1
        self.jobs.put(job, block=True, timeout=HUGE_TIMEOUT)
        logger.info('added a new job (len(queue)=%i items)', self.jobs.qsize())

    @Pyro4.expose
    def getstate(self):
        if False:
            return 10
        'Merge projections from across all workers and get the final projection.\n\n        Returns\n        -------\n        :class:`~gensim.models.lsimodel.Projection`\n            The current projection of the total model.\n\n        '
        logger.info('end of input, assigning all remaining jobs')
        logger.debug('jobs done: %s, jobs received: %s', self._jobsdone, self._jobsreceived)
        while self._jobsdone < self._jobsreceived:
            time.sleep(0.5)
        logger.info('merging states from %i workers', len(self.workers))
        workers = list(self.workers.items())
        result = workers[0][1].getstate()
        for (workerid, worker) in workers[1:]:
            logger.info('pulling state from worker %s', workerid)
            result.merge(worker.getstate())
        logger.info('sending out merged projection')
        return result

    @Pyro4.expose
    def reset(self):
        if False:
            return 10
        'Re-initialize all workers for a new decomposition.'
        for (workerid, worker) in self.workers.items():
            logger.info('resetting worker %s', workerid)
            worker.reset()
            worker.requestjob()
        self._jobsdone = 0
        self._jobsreceived = 0

    @Pyro4.expose
    @Pyro4.oneway
    @utils.synchronous('lock_update')
    def jobdone(self, workerid):
        if False:
            for i in range(10):
                print('nop')
        'A worker has finished its job. Log this event and then asynchronously transfer control back to the worker.\n\n        Callback used by workers to notify when their job is done.\n\n        The job done event is logged and then control is asynchronously transfered back to the worker\n        (who can then request another job). In this way, control flow basically oscillates between\n        :meth:`gensim.models.lsi_dispatcher.Dispatcher.jobdone` and :meth:`gensim.models.lsi_worker.Worker.requestjob`.\n\n        Parameters\n        ----------\n        workerid : int\n            The ID of the worker that finished the job (used for logging).\n\n        '
        self._jobsdone += 1
        logger.info('worker #%s finished job #%i', workerid, self._jobsdone)
        worker = self.workers[workerid]
        worker.requestjob()

    def jobsdone(self):
        if False:
            for i in range(10):
                print('nop')
        'Wrap :attr:`~gensim.models.lsi_dispatcher.Dispatcher._jobsdone`, needed for remote access through proxies.\n\n        Returns\n        -------\n        int\n            Number of jobs already completed.\n\n        '
        return self._jobsdone

    @Pyro4.oneway
    def exit(self):
        if False:
            while True:
                i = 10
        'Terminate all registered workers and then the dispatcher.'
        for (workerid, worker) in self.workers.items():
            logger.info('terminating worker %s', workerid)
            worker.exit()
        logger.info('terminating dispatcher')
        os._exit(0)
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__[:-135], formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('maxsize', nargs='?', type=int, help='Maximum number of jobs to be kept pre-fetched in the queue.', default=MAX_JOBS_QUEUE)
    args = parser.parse_args()
    logger.info('running %s', ' '.join(sys.argv))
    utils.pyro_daemon('gensim.lsi_dispatcher', Dispatcher(maxsize=args.maxsize))
    logger.info('finished running %s', parser.prog)