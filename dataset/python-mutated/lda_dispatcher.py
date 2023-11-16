"""Dispatcher process which orchestrates distributed Latent Dirichlet Allocation
(LDA, :class:`~gensim.models.ldamodel.LdaModel`) computations.
Run this script only once, on any node in your cluster.

Notes
-----
The dispatcher expects to find worker scripts already running. Make sure you run as many workers as you like on
your machines **before** launching the dispatcher.


How to use distributed :class:`~gensim.models.ldamodel.LdaModel`
----------------------------------------------------------------

#. Install needed dependencies (Pyro4) ::

    pip install gensim[distributed]

#. Setup serialization (on each machine) ::

    export PYRO_SERIALIZERS_ACCEPTED=pickle
    export PYRO_SERIALIZER=pickle

#. Run nameserver ::

    python -m Pyro4.naming -n 0.0.0.0 &

#. Run workers (on each machine) ::

    python -m gensim.models.lda_worker &

#. Run dispatcher ::

    python -m gensim.models.lda_dispatcher &

#. Run :class:`~gensim.models.ldamodel.LdaModel` in distributed mode :

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_corpus, common_dictionary
    >>> from gensim.models import LdaModel
    >>>
    >>> model = LdaModel(common_corpus, id2word=common_dictionary, distributed=True)


Command line arguments
----------------------

.. program-output:: python -m gensim.models.lda_dispatcher --help
   :ellipsis: 0, -7

"""
import argparse
import os
import sys
import logging
import threading
import time
from queue import Queue
import Pyro4
from gensim import utils
from gensim.models.lda_worker import LDA_WORKER_PREFIX
logger = logging.getLogger('gensim.models.lda_dispatcher')
MAX_JOBS_QUEUE = 10
HUGE_TIMEOUT = 365 * 24 * 60 * 60
LDA_DISPATCHER_PREFIX = 'gensim.lda_dispatcher'

class Dispatcher:
    """Dispatcher object that communicates and coordinates individual workers.

    Warnings
    --------
    There should never be more than one dispatcher running at any one time.

    """

    def __init__(self, maxsize=MAX_JOBS_QUEUE, ns_conf=None):
        if False:
            i = 10
            return i + 15
        'Partly initializes the dispatcher.\n\n        A full initialization (including initialization of the workers) requires a call to\n        :meth:`~gensim.models.lda_dispatcher.Dispatcher.initialize`\n\n        Parameters\n        ----------\n        maxsize : int, optional\n                Maximum number of jobs to be kept pre-fetched in the queue.\n        ns_conf : dict of (str, object)\n            Sets up the name server configuration for the pyro daemon server of dispatcher.\n            This also helps to keep track of your objects in your network by using logical object names\n            instead of exact object name(or id) and its location.\n\n        '
        self.maxsize = maxsize
        self.callback = None
        self.ns_conf = ns_conf if ns_conf is not None else {}

    @Pyro4.expose
    def initialize(self, **model_params):
        if False:
            i = 10
            return i + 15
        'Fully initialize the dispatcher and all its workers.\n\n        Parameters\n        ----------\n        **model_params\n            Keyword parameters used to initialize individual workers, see :class:`~gensim.models.ldamodel.LdaModel`.\n\n        Raises\n        ------\n        RuntimeError\n            When no workers are found (the :mod:`gensim.models.lda_worker` script must be ran beforehand).\n\n        '
        self.jobs = Queue(maxsize=self.maxsize)
        self.lock_update = threading.Lock()
        self._jobsdone = 0
        self._jobsreceived = 0
        self.workers = {}
        with utils.getNS(**self.ns_conf) as ns:
            self.callback = Pyro4.Proxy(ns.list(prefix=LDA_DISPATCHER_PREFIX)[LDA_DISPATCHER_PREFIX])
            for (name, uri) in ns.list(prefix=LDA_WORKER_PREFIX).items():
                try:
                    worker = Pyro4.Proxy(uri)
                    workerid = len(self.workers)
                    logger.info('registering worker #%i at %s', workerid, uri)
                    worker.initialize(workerid, dispatcher=self.callback, **model_params)
                    self.workers[workerid] = worker
                except Pyro4.errors.PyroError:
                    logger.warning('unresponsive worker at %s,deleting it from the name server', uri)
                    ns.remove(name)
        if not self.workers:
            raise RuntimeError('no workers found; run some lda_worker scripts on your machines first!')

    @Pyro4.expose
    def getworkers(self):
        if False:
            for i in range(10):
                print('nop')
        'Return pyro URIs of all registered workers.\n\n        Returns\n        -------\n        list of URIs\n            The pyro URIs for each worker.\n\n        '
        return [worker._pyroUri for worker in self.workers.values()]

    @Pyro4.expose
    def getjob(self, worker_id):
        if False:
            while True:
                i = 10
        'Atomically pop a job from the queue.\n\n        Parameters\n        ----------\n        worker_id : int\n            The worker that requested the job.\n\n        Returns\n        -------\n        iterable of list of (int, float)\n            The corpus in BoW format.\n\n        '
        logger.info('worker #%i requesting a new job', worker_id)
        job = self.jobs.get(block=True, timeout=1)
        logger.info('worker #%i got a new job (%i left)', worker_id, self.jobs.qsize())
        return job

    @Pyro4.expose
    def putjob(self, job):
        if False:
            while True:
                i = 10
        'Atomically add a job to the queue.\n\n        Parameters\n        ----------\n        job : iterable of list of (int, float)\n            The corpus in BoW format.\n\n        '
        self._jobsreceived += 1
        self.jobs.put(job, block=True, timeout=HUGE_TIMEOUT)
        logger.info('added a new job (len(queue)=%i items)', self.jobs.qsize())

    @Pyro4.expose
    def getstate(self):
        if False:
            for i in range(10):
                print('nop')
        'Merge states from across all workers and return the result.\n\n        Returns\n        -------\n        :class:`~gensim.models.ldamodel.LdaState`\n            Merged resultant state\n\n        '
        logger.info('end of input, assigning all remaining jobs')
        logger.debug('jobs done: %s, jobs received: %s', self._jobsdone, self._jobsreceived)
        i = 0
        count = 10
        while self._jobsdone < self._jobsreceived:
            time.sleep(0.5)
            i += 1
            if i > count:
                i = 0
                for (workerid, worker) in self.workers.items():
                    logger.info('checking aliveness for worker %s', workerid)
                    worker.ping()
        logger.info('merging states from %i workers', len(self.workers))
        workers = list(self.workers.values())
        result = workers[0].getstate()
        for worker in workers[1:]:
            result.merge(worker.getstate())
        logger.info('sending out merged state')
        return result

    @Pyro4.expose
    def reset(self, state):
        if False:
            i = 10
            return i + 15
        'Reinitialize all workers for a new EM iteration.\n\n        Parameters\n        ----------\n        state : :class:`~gensim.models.ldamodel.LdaState`\n            State of :class:`~gensim.models.lda.LdaModel`.\n\n        '
        for (workerid, worker) in self.workers.items():
            logger.info('resetting worker %s', workerid)
            worker.reset(state)
            worker.requestjob()
        self._jobsdone = 0
        self._jobsreceived = 0

    @Pyro4.expose
    @Pyro4.oneway
    @utils.synchronous('lock_update')
    def jobdone(self, workerid):
        if False:
            i = 10
            return i + 15
        'A worker has finished its job. Log this event and then asynchronously transfer control back to the worker.\n\n        Callback used by workers to notify when their job is done.\n\n        The job done event is logged and then control is asynchronously transfered back to the worker\n        (who can then request another job). In this way, control flow basically oscillates between\n        :meth:`gensim.models.lda_dispatcher.Dispatcher.jobdone` and :meth:`gensim.models.lda_worker.Worker.requestjob`.\n\n        Parameters\n        ----------\n        workerid : int\n            The ID of the worker that finished the job (used for logging).\n\n        '
        self._jobsdone += 1
        logger.info('worker #%s finished job #%i', workerid, self._jobsdone)
        self.workers[workerid].requestjob()

    def jobsdone(self):
        if False:
            return 10
        'Wrap :attr:`~gensim.models.lda_dispatcher.Dispatcher._jobsdone` needed for remote access through proxies.\n\n        Returns\n        -------\n        int\n            Number of jobs already completed.\n\n        '
        return self._jobsdone

    @Pyro4.oneway
    def exit(self):
        if False:
            print('Hello World!')
        'Terminate all registered workers and then the dispatcher.'
        for (workerid, worker) in self.workers.items():
            logger.info('terminating worker %s', workerid)
            worker.exit()
        logger.info('terminating dispatcher')
        os._exit(0)

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description=__doc__[:-135], formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--maxsize', help="How many jobs (=chunks of N documents) to keep 'pre-fetched' in a queue (default: %(default)s)", type=int, default=MAX_JOBS_QUEUE)
    parser.add_argument('--host', help='Nameserver hostname (default: %(default)s)', default=None)
    parser.add_argument('--port', help='Nameserver port (default: %(default)s)', default=None, type=int)
    parser.add_argument('--no-broadcast', help='Disable broadcast (default: %(default)s)', action='store_const', default=True, const=False)
    parser.add_argument('--hmac', help='Nameserver hmac key (default: %(default)s)', default=None)
    parser.add_argument('-v', '--verbose', help='Verbose flag', action='store_const', dest='loglevel', const=logging.INFO, default=logging.WARNING)
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=args.loglevel)
    logger.info('running %s', ' '.join(sys.argv))
    ns_conf = {'broadcast': args.no_broadcast, 'host': args.host, 'port': args.port, 'hmac_key': args.hmac}
    utils.pyro_daemon(LDA_DISPATCHER_PREFIX, Dispatcher(maxsize=args.maxsize, ns_conf=ns_conf), ns_conf=ns_conf)
    logger.info('finished running %s', ' '.join(sys.argv))
if __name__ == '__main__':
    main()