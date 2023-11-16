"""Worker ("slave") process used in computing distributed Latent Dirichlet Allocation
(LDA, :class:`~gensim.models.ldamodel.LdaModel`).

Run this script on every node in your cluster. If you wish, you may even run it multiple times on a single machine,
to make better use of multiple cores (just beware that memory footprint increases accordingly).


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

.. program-output:: python -m gensim.models.lda_worker --help
   :ellipsis: 0, -7

"""
from __future__ import with_statement
import os
import sys
import logging
import threading
import tempfile
import argparse
try:
    import Queue
except ImportError:
    import queue as Queue
import Pyro4
from gensim.models import ldamodel
from gensim import utils
logger = logging.getLogger('gensim.models.lda_worker')
SAVE_DEBUG = 0
LDA_WORKER_PREFIX = 'gensim.lda_worker'

class Worker:
    """Used as a Pyro4 class with exposed methods.

    Exposes every non-private method and property of the class automatically to be available for remote access.

    """

    def __init__(self):
        if False:
            print('Hello World!')
        'Partly initialize the model.'
        self.model = None

    @Pyro4.expose
    def initialize(self, myid, dispatcher, **model_params):
        if False:
            return 10
        'Fully initialize the worker.\n\n        Parameters\n        ----------\n        myid : int\n            An ID number used to identify this worker in the dispatcher object.\n        dispatcher : :class:`~gensim.models.lda_dispatcher.Dispatcher`\n            The dispatcher responsible for scheduling this worker.\n        **model_params\n            Keyword parameters to initialize the inner LDA model,see :class:`~gensim.models.ldamodel.LdaModel`.\n\n        '
        self.lock_update = threading.Lock()
        self.jobsdone = 0
        self.myid = myid
        self.dispatcher = dispatcher
        self.finished = False
        logger.info('initializing worker #%s', myid)
        self.model = ldamodel.LdaModel(**model_params)

    @Pyro4.expose
    @Pyro4.oneway
    def requestjob(self):
        if False:
            for i in range(10):
                print('nop')
        'Request jobs from the dispatcher, in a perpetual loop until :meth:`gensim.models.lda_worker.Worker.getstate`\n        is called.\n\n        Raises\n        ------\n        RuntimeError\n            If `self.model` is None (i.e. worker non initialized).\n\n        '
        if self.model is None:
            raise RuntimeError('worker must be initialized before receiving jobs')
        job = None
        while job is None and (not self.finished):
            try:
                job = self.dispatcher.getjob(self.myid)
            except Queue.Empty:
                continue
        if job is not None:
            logger.info('worker #%s received job #%i', self.myid, self.jobsdone)
            self.processjob(job)
            self.dispatcher.jobdone(self.myid)
        else:
            logger.info('worker #%i stopping asking for jobs', self.myid)

    @utils.synchronous('lock_update')
    def processjob(self, job):
        if False:
            for i in range(10):
                print('nop')
        'Incrementally process the job and potentially logs progress.\n\n        Parameters\n        ----------\n        job : iterable of list of (int, float)\n            Corpus in BoW format.\n\n        '
        logger.debug('starting to process job #%i', self.jobsdone)
        self.model.do_estep(job)
        self.jobsdone += 1
        if SAVE_DEBUG and self.jobsdone % SAVE_DEBUG == 0:
            fname = os.path.join(tempfile.gettempdir(), 'lda_worker.pkl')
            self.model.save(fname)
        logger.info('finished processing job #%i', self.jobsdone - 1)

    @Pyro4.expose
    def ping(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the connectivity with Worker.'
        return True

    @Pyro4.expose
    @utils.synchronous('lock_update')
    def getstate(self):
        if False:
            while True:
                i = 10
        "Log and get the LDA model's current state.\n\n        Returns\n        -------\n        result : :class:`~gensim.models.ldamodel.LdaState`\n            The current state.\n\n        "
        logger.info('worker #%i returning its state after %s jobs', self.myid, self.jobsdone)
        result = self.model.state
        assert isinstance(result, ldamodel.LdaState)
        self.model.clear()
        self.finished = True
        return result

    @Pyro4.expose
    @utils.synchronous('lock_update')
    def reset(self, state):
        if False:
            return 10
        'Reset the worker by setting sufficient stats to 0.\n\n        Parameters\n        ----------\n        state : :class:`~gensim.models.ldamodel.LdaState`\n            Encapsulates information for distributed computation of LdaModel objects.\n\n        '
        assert state is not None
        logger.info('resetting worker #%i', self.myid)
        self.model.state = state
        self.model.sync_state()
        self.model.state.reset()
        self.finished = False

    @Pyro4.oneway
    def exit(self):
        if False:
            for i in range(10):
                print('nop')
        'Terminate the worker.'
        logger.info('terminating worker #%i', self.myid)
        os._exit(0)

def main():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description=__doc__[:-130], formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--host', help='Nameserver hostname (default: %(default)s)', default=None)
    parser.add_argument('--port', help='Nameserver port (default: %(default)s)', default=None, type=int)
    parser.add_argument('--no-broadcast', help='Disable broadcast (default: %(default)s)', action='store_const', default=True, const=False)
    parser.add_argument('--hmac', help='Nameserver hmac key (default: %(default)s)', default=None)
    parser.add_argument('-v', '--verbose', help='Verbose flag', action='store_const', dest='loglevel', const=logging.INFO, default=logging.WARNING)
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=args.loglevel)
    logger.info('running %s', ' '.join(sys.argv))
    ns_conf = {'broadcast': args.no_broadcast, 'host': args.host, 'port': args.port, 'hmac_key': args.hmac}
    utils.pyro_daemon(LDA_WORKER_PREFIX, Worker(), random_suffix=True, ns_conf=ns_conf)
    logger.info('finished running %s', ' '.join(sys.argv))
if __name__ == '__main__':
    main()