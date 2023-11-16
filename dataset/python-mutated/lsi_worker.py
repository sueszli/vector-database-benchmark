"""Worker ("slave") process used in computing distributed Latent Semantic Indexing (LSI,
:class:`~gensim.models.lsimodel.LsiModel`) models.

Run this script on every node in your cluster. If you wish, you may even run it multiple times on a single machine,
to make better use of multiple cores (just beware that memory footprint increases linearly).


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

.. program-output:: python -m gensim.models.lsi_worker --help
   :ellipsis: 0, -3

"""
import os
import sys
import logging
import argparse
import threading
import tempfile
import queue as Queue
import Pyro4
from gensim.models import lsimodel
from gensim import utils
logger = logging.getLogger(__name__)
SAVE_DEBUG = 0

class Worker:

    def __init__(self):
        if False:
            print('Hello World!')
        'Partly initialize the model.\n\n        A full initialization requires a call to :meth:`~gensim.models.lsi_worker.Worker.initialize`.\n\n        '
        self.model = None

    @Pyro4.expose
    def initialize(self, myid, dispatcher, **model_params):
        if False:
            while True:
                i = 10
        'Fully initialize the worker.\n\n        Parameters\n        ----------\n        myid : int\n            An ID number used to identify this worker in the dispatcher object.\n        dispatcher : :class:`~gensim.models.lsi_dispatcher.Dispatcher`\n            The dispatcher responsible for scheduling this worker.\n        **model_params\n            Keyword parameters to initialize the inner LSI model, see :class:`~gensim.models.lsimodel.LsiModel`.\n\n        '
        self.lock_update = threading.Lock()
        self.jobsdone = 0
        self.myid = myid
        self.dispatcher = dispatcher
        self.finished = False
        logger.info('initializing worker #%s', myid)
        self.model = lsimodel.LsiModel(**model_params)

    @Pyro4.expose
    @Pyro4.oneway
    def requestjob(self):
        if False:
            return 10
        'Request jobs from the dispatcher, in a perpetual loop until :meth:`~gensim.models.lsi_worker.Worker.getstate`\n        is called.\n\n        Raises\n        ------\n        RuntimeError\n            If `self.model` is None (i.e. worker not initialized).\n\n        '
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
        self.model.add_documents(job)
        self.jobsdone += 1
        if SAVE_DEBUG and self.jobsdone % SAVE_DEBUG == 0:
            fname = os.path.join(tempfile.gettempdir(), 'lsi_worker.pkl')
            self.model.save(fname)

    @Pyro4.expose
    @utils.synchronous('lock_update')
    def getstate(self):
        if False:
            for i in range(10):
                print('nop')
        "Log and get the LSI model's current projection.\n\n        Returns\n        -------\n        :class:`~gensim.models.lsimodel.Projection`\n            The current projection.\n\n        "
        logger.info('worker #%i returning its state after %s jobs', self.myid, self.jobsdone)
        assert isinstance(self.model.projection, lsimodel.Projection)
        self.finished = True
        return self.model.projection

    @Pyro4.expose
    @utils.synchronous('lock_update')
    def reset(self):
        if False:
            print('Hello World!')
        'Reset the worker by deleting its current projection.'
        logger.info('resetting worker #%i', self.myid)
        self.model.projection = self.model.projection.empty_like()
        self.finished = False

    @Pyro4.oneway
    def exit(self):
        if False:
            print('Hello World!')
        'Terminate the worker.'
        logger.info('terminating worker #%i', self.myid)
        os._exit(0)
if __name__ == '__main__':
    'The main script. '
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__[:-135], formatter_class=argparse.RawTextHelpFormatter)
    _ = parser.parse_args()
    logger.info('running %s', ' '.join(sys.argv))
    utils.pyro_daemon('gensim.lsi_worker', Worker(), random_suffix=True)
    logger.info('finished running %s', parser.prog)