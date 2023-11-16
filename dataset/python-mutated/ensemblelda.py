"""Ensemble Latent Dirichlet Allocation (eLDA), an algorithm for extracting reliable topics.

The aim of topic modelling is to find a set of topics that represent the global structure of a corpus of documents. One
issue that occurs with topics extracted from an NMF or LDA model is reproducibility. That is, if the topic model is
trained repeatedly allowing only the random seed to change, would the same (or similar) topic representation be reliably
learned. Unreliable topics are undesireable because they are not a good representation of the corpus.

Ensemble LDA addresses the issue by training an ensemble of topic models and throwing out topics that do not reoccur
across the ensemble. In this regard, the topics extracted are more reliable and there is the added benefit over many
topic models that the user does not need to know the exact number of topics ahead of time.

For more information, see the :ref:`citation section <Citation>` below, watch our `Machine Learning Prague 2019 talk
<https://slideslive.com/38913528/solving-the-text-labeling-challenge-with-ensemblelda-and-active-learning?locale=cs>`_,
or view our `Machine Learning Summer School poster
<https://github.com/aloosley/ensembleLDA/blob/master/mlss/mlss_poster_v2.pdf>`_.

Usage examples
--------------

Train an ensemble of LdaModels using a Gensim corpus:

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_texts
    >>> from gensim.corpora.dictionary import Dictionary
    >>> from gensim.models import EnsembleLda
    >>>
    >>> # Create a corpus from a list of texts
    >>> common_dictionary = Dictionary(common_texts)
    >>> common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
    >>>
    >>> # Train the model on the corpus. corpus has to be provided as a
    >>> # keyword argument, as they are passed through to the children.
    >>> elda = EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=10, num_models=4)

Save a model to disk, or reload a pre-trained model:

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>>
    >>> # Save model to disk.
    >>> temp_file = datapath("model")
    >>> elda.save(temp_file)
    >>>
    >>> # Load a potentially pretrained model from disk.
    >>> elda = EnsembleLda.load(temp_file)

Query, the model using new, unseen documents:

.. sourcecode:: pycon

    >>> # Create a new corpus, made of previously unseen documents.
    >>> other_texts = [
    ...     ['computer', 'time', 'graph'],
    ...     ['survey', 'response', 'eps'],
    ...     ['human', 'system', 'computer']
    ... ]
    >>> other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]
    >>>
    >>> unseen_doc = other_corpus[0]
    >>> vector = elda[unseen_doc]  # get topic probability distribution for a document

Increase the ensemble size by adding a new model. Make sure it uses the same dictionary:

.. sourcecode:: pycon

    >>> from gensim.models import LdaModel
    >>> elda.add_model(LdaModel(common_corpus, id2word=common_dictionary, num_topics=10))
    >>> elda.recluster()
    >>> vector = elda[unseen_doc]

To optimize the ensemble for your specific case, the children can be clustered again using
different hyperparameters:

.. sourcecode:: pycon

    >>> elda.recluster(eps=0.2)

.. _Citation:

Citation
--------
BRIGL, Tobias, 2019, Extracting Reliable Topics using Ensemble Latent Dirichlet Allocation [Bachelor Thesis].
Technische Hochschule Ingolstadt. Munich: Data Reply GmbH. Supervised by Alex Loosley. Available from:
https://www.sezanzeb.de/machine_learning/ensemble_LDA/

"""
import logging
import os
from multiprocessing import Process, Pipe, ProcessError
import importlib
from typing import Set, Optional, List
import numpy as np
from scipy.spatial.distance import cosine
from dataclasses import dataclass
from gensim import utils
from gensim.models import ldamodel, ldamulticore, basemodel
from gensim.utils import SaveLoad
logger = logging.getLogger(__name__)
_COSINE_DISTANCE_CALCULATION_THRESHOLD = 0.05
_MAX_RANDOM_STATE = np.iinfo(np.int32).max

@dataclass
class Topic:
    is_core: bool
    neighboring_labels: Set[int]
    neighboring_topic_indices: Set[int]
    label: Optional[int]
    num_neighboring_labels: int
    valid_neighboring_labels: Set[int]

@dataclass
class Cluster:
    max_num_neighboring_labels: int
    neighboring_labels: List[Set[int]]
    label: int
    num_cores: int

def _is_valid_core(topic):
    if False:
        for i in range(10):
            print('nop')
    'Check if the topic is a valid core, i.e. no neighboring valid cluster is overlapping with it.\n\n    Parameters\n    ----------\n    topic : :class:`Topic`\n        topic to validate\n\n    '
    return topic.is_core and topic.valid_neighboring_labels == {topic.label}

def _remove_from_all_sets(label, clusters):
    if False:
        return 10
    'Remove a label from every set in "neighboring_labels" for each core in ``clusters``.'
    for cluster in clusters:
        for neighboring_labels_set in cluster.neighboring_labels:
            if label in neighboring_labels_set:
                neighboring_labels_set.remove(label)

def _contains_isolated_cores(label, cluster, min_cores):
    if False:
        while True:
            i = 10
    'Check if the cluster has at least ``min_cores`` of cores that belong to no other cluster.'
    return sum([neighboring_labels == {label} for neighboring_labels in cluster.neighboring_labels]) >= min_cores

def _aggregate_topics(grouped_by_labels):
    if False:
        while True:
            i = 10
    'Aggregate the labeled topics to a list of clusters.\n\n    Parameters\n    ----------\n    grouped_by_labels : dict of (int, list of :class:`Topic`)\n        The return value of _group_by_labels. A mapping of the label to a list of each topic which belongs to the\n        label.\n\n    Returns\n    -------\n    list of :class:`Cluster`\n        It is sorted by max_num_neighboring_labels in descending order. There is one single element for each cluster.\n\n    '
    clusters = []
    for (label, topics) in grouped_by_labels.items():
        max_num_neighboring_labels = 0
        neighboring_labels = []
        for topic in topics:
            max_num_neighboring_labels = max(topic.num_neighboring_labels, max_num_neighboring_labels)
            neighboring_labels.append(topic.neighboring_labels)
        neighboring_labels = [x for x in neighboring_labels if len(x) > 0]
        clusters.append(Cluster(max_num_neighboring_labels=max_num_neighboring_labels, neighboring_labels=neighboring_labels, label=label, num_cores=len([topic for topic in topics if topic.is_core])))
    logger.info('found %s clusters', len(clusters))
    return clusters

def _group_by_labels(cbdbscan_topics):
    if False:
        for i in range(10):
            print('nop')
    'Group all the learned cores by their label, which was assigned in the cluster_model.\n\n    Parameters\n    ----------\n    cbdbscan_topics : list of :class:`Topic`\n        A list of topic data resulting from fitting a :class:`~CBDBSCAN` object.\n        After calling .fit on a CBDBSCAN model, the results can be retrieved from it by accessing the .results\n        member, which can be used as the argument to this function. It is a list of infos gathered during\n        the clustering step and each element in the list corresponds to a single topic.\n\n    Returns\n    -------\n    dict of (int, list of :class:`Topic`)\n        A mapping of the label to a list of topics that belong to that particular label. Also adds\n        a new member to each topic called num_neighboring_labels, which is the number of\n        neighboring_labels of that topic.\n\n    '
    grouped_by_labels = {}
    for topic in cbdbscan_topics:
        if topic.is_core:
            topic.num_neighboring_labels = len(topic.neighboring_labels)
            label = topic.label
            if label not in grouped_by_labels:
                grouped_by_labels[label] = []
            grouped_by_labels[label].append(topic)
    return grouped_by_labels

def _teardown(pipes, processes, i):
    if False:
        while True:
            i = 10
    'Close pipes and terminate processes.\n\n    Parameters\n    ----------\n        pipes : {list of :class:`multiprocessing.Pipe`}\n            list of pipes that the processes use to communicate with the parent\n        processes : {list of :class:`multiprocessing.Process`}\n            list of worker processes\n    '
    for (parent_conn, child_conn) in pipes:
        child_conn.close()
        parent_conn.close()
    for process in processes:
        if process.is_alive():
            process.terminate()
        del process

def mass_masking(a, threshold=None):
    if False:
        return 10
    'Original masking method. Returns a new binary mask.'
    if threshold is None:
        threshold = 0.95
    sorted_a = np.sort(a)[::-1]
    largest_mass = sorted_a.cumsum() < threshold
    smallest_valid = sorted_a[largest_mass][-1]
    return a >= smallest_valid

def rank_masking(a, threshold=None):
    if False:
        for i in range(10):
            print('nop')
    'Faster masking method. Returns a new binary mask.'
    if threshold is None:
        threshold = 0.11
    return a > np.sort(a)[::-1][int(len(a) * threshold)]

def _validate_clusters(clusters, min_cores):
    if False:
        for i in range(10):
            print('nop')
    'Check which clusters from the cbdbscan step are significant enough. is_valid is set accordingly.'

    def _cluster_sort_key(cluster):
        if False:
            i = 10
            return i + 15
        return (cluster.max_num_neighboring_labels, cluster.num_cores, cluster.label)
    sorted_clusters = sorted(clusters, key=_cluster_sort_key, reverse=False)
    for cluster in sorted_clusters:
        cluster.is_valid = None
        if cluster.num_cores < min_cores:
            cluster.is_valid = False
            _remove_from_all_sets(cluster.label, sorted_clusters)
    for cluster in [cluster for cluster in sorted_clusters if cluster.is_valid is None]:
        label = cluster.label
        if _contains_isolated_cores(label, cluster, min_cores):
            cluster.is_valid = True
        else:
            cluster.is_valid = False
            _remove_from_all_sets(label, sorted_clusters)
    return [cluster for cluster in sorted_clusters if cluster.is_valid]

def _generate_topic_models_multiproc(ensemble, num_models, ensemble_workers):
    if False:
        return 10
    'Generate the topic models to form the ensemble in a multiprocessed way.\n\n    Depending on the used topic model this can result in a speedup.\n\n    Parameters\n    ----------\n    ensemble: EnsembleLda\n        the ensemble\n    num_models : int\n        how many models to train in the ensemble\n    ensemble_workers : int\n        into how many processes to split the models will be set to max(workers, num_models), to avoid workers that\n        are supposed to train 0 models.\n\n        to get maximum performance, set to the number of your cores, if non-parallelized models are being used in\n        the ensemble (LdaModel).\n\n        For LdaMulticore, the performance gain is small and gets larger for a significantly smaller corpus.\n        In that case, ensemble_workers=2 can be used.\n\n    '
    random_states = [ensemble.random_state.randint(_MAX_RANDOM_STATE) for _ in range(num_models)]
    workers = min(ensemble_workers, num_models)
    processes = []
    pipes = []
    num_models_unhandled = num_models
    for i in range(workers):
        (parent_conn, child_conn) = Pipe()
        num_subprocess_models = 0
        if i == workers - 1:
            num_subprocess_models = num_models_unhandled
        else:
            num_subprocess_models = int(num_models_unhandled / (workers - i))
        random_states_for_worker = random_states[-num_models_unhandled:][:num_subprocess_models]
        args = (ensemble, num_subprocess_models, random_states_for_worker, child_conn)
        try:
            process = Process(target=_generate_topic_models_worker, args=args)
            processes.append(process)
            pipes.append((parent_conn, child_conn))
            process.start()
            num_models_unhandled -= num_subprocess_models
        except ProcessError:
            logger.error(f'could not start process {i}')
            _teardown(pipes, processes)
            raise
    for (parent_conn, _) in pipes:
        answer = parent_conn.recv()
        parent_conn.close()
        if not ensemble.memory_friendly_ttda:
            ensemble.tms += answer
            ttda = np.concatenate([m.get_topics() for m in answer])
        else:
            ttda = answer
        ensemble.ttda = np.concatenate([ensemble.ttda, ttda])
    for process in processes:
        process.terminate()

def _generate_topic_models(ensemble, num_models, random_states=None):
    if False:
        i = 10
        return i + 15
    'Train the topic models that form the ensemble.\n\n    Parameters\n    ----------\n    ensemble: EnsembleLda\n        the ensemble\n    num_models : int\n        number of models to be generated\n    random_states : list\n        list of numbers or np.random.RandomState objects. Will be autogenerated based on the ensembles\n        RandomState if None (default).\n    '
    if random_states is None:
        random_states = [ensemble.random_state.randint(_MAX_RANDOM_STATE) for _ in range(num_models)]
    assert len(random_states) == num_models
    kwargs = ensemble.gensim_kw_args.copy()
    tm = None
    for i in range(num_models):
        kwargs['random_state'] = random_states[i]
        tm = ensemble.get_topic_model_class()(**kwargs)
        ensemble.ttda = np.concatenate([ensemble.ttda, tm.get_topics()])
        if not ensemble.memory_friendly_ttda:
            ensemble.tms += [tm]
    ensemble.sstats_sum = tm.state.sstats.sum()
    ensemble.eta = tm.eta

def _generate_topic_models_worker(ensemble, num_models, random_states, pipe):
    if False:
        for i in range(10):
            print('nop')
    'Wrapper for _generate_topic_models to write the results into a pipe.\n\n    This is intended to be used inside a subprocess.'
    logger.info(f'spawned worker to generate {num_models} topic models')
    _generate_topic_models(ensemble=ensemble, num_models=num_models, random_states=random_states)
    if ensemble.memory_friendly_ttda:
        pipe.send(ensemble.ttda)
    else:
        pipe.send(ensemble.tms)
    pipe.close()

def _calculate_asymmetric_distance_matrix_chunk(ttda1, ttda2, start_index, masking_method, masking_threshold):
    if False:
        i = 10
        return i + 15
    'Calculate an (asymmetric) distance from each topic in ``ttda1`` to each topic in ``ttda2``.\n\n    Parameters\n    ----------\n    ttda1 and ttda2: 2D arrays of floats\n        Two ttda matrices that are going to be used for distance calculation. Each row in ttda corresponds to one\n        topic. Each cell in the resulting matrix corresponds to the distance between a topic pair.\n    start_index : int\n        this function might be used in multiprocessing, so start_index has to be set as ttda1 is a chunk of the\n        complete ttda in that case. start_index would be 0 if ``ttda1 == self.ttda``. When self.ttda is split into\n        two pieces, each 100 ttdas long, then start_index should be be 100. default is 0\n    masking_method: function\n\n    masking_threshold: float\n\n    Returns\n    -------\n    2D numpy.ndarray of floats\n        Asymmetric distance matrix of size ``len(ttda1)`` by ``len(ttda2)``.\n\n    '
    distances = np.ndarray((len(ttda1), len(ttda2)))
    if ttda1.shape[0] > 0 and ttda2.shape[0] > 0:
        avg_mask_size = 0
        for (ttd1_idx, ttd1) in enumerate(ttda1):
            mask = masking_method(ttd1, masking_threshold)
            ttd1_masked = ttd1[mask]
            avg_mask_size += mask.sum()
            for (ttd2_idx, ttd2) in enumerate(ttda2):
                if ttd1_idx + start_index == ttd2_idx:
                    distances[ttd1_idx][ttd2_idx] = 0
                    continue
                ttd2_masked = ttd2[mask]
                if ttd2_masked.sum() <= _COSINE_DISTANCE_CALCULATION_THRESHOLD:
                    distance = 1
                else:
                    distance = cosine(ttd1_masked, ttd2_masked)
                distances[ttd1_idx][ttd2_idx] = distance
        percent = round(100 * avg_mask_size / ttda1.shape[0] / ttda1.shape[1], 1)
        logger.info(f'the given threshold of {masking_threshold} covered on average {percent}% of tokens')
    return distances

def _asymmetric_distance_matrix_worker(worker_id, entire_ttda, ttdas_sent, n_ttdas, masking_method, masking_threshold, pipe):
    if False:
        i = 10
        return i + 15
    'Worker that computes the distance to all other nodes from a chunk of nodes.'
    logger.info(f'spawned worker {worker_id} to generate {n_ttdas} rows of the asymmetric distance matrix')
    ttda1 = entire_ttda[ttdas_sent:ttdas_sent + n_ttdas]
    distance_chunk = _calculate_asymmetric_distance_matrix_chunk(ttda1=ttda1, ttda2=entire_ttda, start_index=ttdas_sent, masking_method=masking_method, masking_threshold=masking_threshold)
    pipe.send((worker_id, distance_chunk))
    pipe.close()

def _calculate_assymetric_distance_matrix_multiproc(workers, entire_ttda, masking_method, masking_threshold):
    if False:
        print('Hello World!')
    processes = []
    pipes = []
    ttdas_sent = 0
    for i in range(workers):
        try:
            (parent_conn, child_conn) = Pipe()
            n_ttdas = 0
            if i == workers - 1:
                n_ttdas = len(entire_ttda) - ttdas_sent
            else:
                n_ttdas = int((len(entire_ttda) - ttdas_sent) / (workers - i))
            args = (i, entire_ttda, ttdas_sent, n_ttdas, masking_method, masking_threshold, child_conn)
            process = Process(target=_asymmetric_distance_matrix_worker, args=args)
            ttdas_sent += n_ttdas
            processes.append(process)
            pipes.append((parent_conn, child_conn))
            process.start()
        except ProcessError:
            logger.error(f'could not start process {i}')
            _teardown(pipes, processes)
            raise
    distances = []
    for (parent_conn, _) in pipes:
        (worker_id, distance_chunk) = parent_conn.recv()
        parent_conn.close()
        distances.append(distance_chunk)
    for process in processes:
        process.terminate()
    return np.concatenate(distances)

class EnsembleLda(SaveLoad):
    """Ensemble Latent Dirichlet Allocation (eLDA), a method of training a topic model ensemble.

    Extracts stable topics that are consistently learned across multiple LDA models. eLDA has the added benefit that
    the user does not need to know the exact number of topics the topic model should extract ahead of time.

    """

    def __init__(self, topic_model_class='ldamulticore', num_models=3, min_cores=None, epsilon=0.1, ensemble_workers=1, memory_friendly_ttda=True, min_samples=None, masking_method=mass_masking, masking_threshold=None, distance_workers=1, random_state=None, **gensim_kw_args):
        if False:
            print('Hello World!')
        'Create and train a new EnsembleLda model.\n\n        Will start training immediatelly, except if iterations, passes or num_models is 0 or if the corpus is missing.\n\n        Parameters\n        ----------\n        topic_model_class : str, topic model, optional\n            Examples:\n                * \'ldamulticore\' (default, recommended)\n                * \'lda\'\n                * ldamodel.LdaModel\n                * ldamulticore.LdaMulticore\n        ensemble_workers : int, optional\n            Spawns that many processes and distributes the models from the ensemble to those as evenly as possible.\n            num_models should be a multiple of ensemble_workers.\n\n            Setting it to 0 or 1 will both use the non-multiprocessing version. Default: 1\n        num_models : int, optional\n            How many LDA models to train in this ensemble.\n            Default: 3\n        min_cores : int, optional\n            Minimum cores a cluster of topics has to contain so that it is recognized as stable topic.\n        epsilon : float, optional\n            Defaults to 0.1. Epsilon for the CBDBSCAN clustering that generates the stable topics.\n        ensemble_workers : int, optional\n            Spawns that many processes and distributes the models from the ensemble to those as evenly as possible.\n            num_models should be a multiple of ensemble_workers.\n\n            Setting it to 0 or 1 will both use the nonmultiprocessing version. Default: 1\n        memory_friendly_ttda : boolean, optional\n            If True, the models in the ensemble are deleted after training and only a concatenation of each model\'s\n            topic term distribution (called ttda) is kept to save memory.\n\n            Defaults to True. When False, trained models are stored in a list in self.tms, and no models that are not\n            of a gensim model type can be added to this ensemble using the add_model function.\n\n            If False, any topic term matrix can be suplied to add_model.\n        min_samples : int, optional\n            Required int of nearby topics for a topic to be considered as \'core\' in the CBDBSCAN clustering.\n        masking_method : function, optional\n            Choose one of :meth:`~gensim.models.ensemblelda.mass_masking` (default) or\n            :meth:`~gensim.models.ensemblelda.rank_masking` (percentile, faster).\n\n            For clustering, distances between topic-term distributions are asymmetric.  In particular, the distance\n            (technically a divergence) from distribution A to B is more of a measure of if A is contained in B.  At a\n            high level, this involves using distribution A to mask distribution B and then calculating the cosine\n            distance between the two.  The masking can be done in two ways:\n\n            1. mass: forms mask by taking the top ranked terms until their cumulative mass reaches the\n            \'masking_threshold\'\n\n            2. rank: forms mask by taking the top ranked terms (by mass) until the \'masking_threshold\' is reached.\n            For example, a ranking threshold of 0.11 means the top 0.11 terms by weight are used to form a mask.\n        masking_threshold : float, optional\n            Default: None, which uses ``0.95`` for "mass", and ``0.11`` for masking_method "rank".  In general, too\n            small a mask threshold leads to inaccurate calculations (no signal) and too big a mask leads to noisy\n            distance calculations.  Defaults are often a good sweet spot for this hyperparameter.\n        distance_workers : int, optional\n            When ``distance_workers`` is ``None``, it defaults to ``os.cpu_count()`` for maximum performance. Default is\n            1, which is not multiprocessed. Set to ``> 1`` to enable multiprocessing.\n        **gensim_kw_args\n            Parameters for each gensim model (e.g. :py:class:`gensim.models.LdaModel`) in the ensemble.\n\n        '
        if 'id2word' not in gensim_kw_args:
            gensim_kw_args['id2word'] = None
        if 'corpus' not in gensim_kw_args:
            gensim_kw_args['corpus'] = None
        if gensim_kw_args['id2word'] is None and (not gensim_kw_args['corpus'] is None):
            logger.warning('no word id mapping provided; initializing from corpus, assuming identity')
            gensim_kw_args['id2word'] = utils.dict_from_corpus(gensim_kw_args['corpus'])
        if gensim_kw_args['id2word'] is None and gensim_kw_args['corpus'] is None:
            raise ValueError('at least one of corpus/id2word must be specified, to establish input space dimensionality. Corpus should be provided using the `corpus` keyword argument.')
        if type(topic_model_class) == type and issubclass(topic_model_class, ldamodel.LdaModel):
            self.topic_model_class = topic_model_class
        else:
            kinds = {'lda': ldamodel.LdaModel, 'ldamulticore': ldamulticore.LdaMulticore}
            if topic_model_class not in kinds:
                raise ValueError("topic_model_class should be one of 'lda', 'ldamulticode' or a model inheriting from LdaModel")
            self.topic_model_class = kinds[topic_model_class]
        self.num_models = num_models
        self.gensim_kw_args = gensim_kw_args
        self.memory_friendly_ttda = memory_friendly_ttda
        self.distance_workers = distance_workers
        self.masking_threshold = masking_threshold
        self.masking_method = masking_method
        self.classic_model_representation = None
        self.random_state = utils.get_random_state(random_state)
        self.sstats_sum = 0
        self.eta = None
        self.tms = []
        self.ttda = np.empty((0, len(gensim_kw_args['id2word'])))
        self.asymmetric_distance_matrix_outdated = True
        if num_models <= 0:
            return
        if gensim_kw_args.get('corpus') is None:
            return
        if 'iterations' in gensim_kw_args and gensim_kw_args['iterations'] <= 0:
            return
        if 'passes' in gensim_kw_args and gensim_kw_args['passes'] <= 0:
            return
        logger.info(f'generating {num_models} topic models using {ensemble_workers} workers')
        if ensemble_workers > 1:
            _generate_topic_models_multiproc(self, num_models, ensemble_workers)
        else:
            _generate_topic_models(self, num_models)
        self._generate_asymmetric_distance_matrix()
        self._generate_topic_clusters(epsilon, min_samples)
        self._generate_stable_topics(min_cores)
        self.generate_gensim_representation()

    def get_topic_model_class(self):
        if False:
            while True:
                i = 10
        'Get the class that is used for :meth:`gensim.models.EnsembleLda.generate_gensim_representation`.'
        if self.topic_model_class is None:
            instruction = 'Try setting topic_model_class manually to what the individual models were based on, e.g. LdaMulticore.'
            try:
                module = importlib.import_module(self.topic_model_module_string)
                self.topic_model_class = getattr(module, self.topic_model_class_string)
                del self.topic_model_module_string
                del self.topic_model_class_string
            except ModuleNotFoundError:
                logger.error(f'Could not import the "{self.topic_model_class_string}" module in order to provide the "{self.topic_model_class_string}" class as "topic_model_class" attribute. {instruction}')
            except AttributeError:
                logger.error(f'Could not import the "{self.topic_model_class_string}" class from the "{self.topic_model_module_string}" module in order to set the "topic_model_class" attribute. {instruction}')
        return self.topic_model_class

    def save(self, *args, **kwargs):
        if False:
            print('Hello World!')
        if self.get_topic_model_class() is not None:
            self.topic_model_module_string = self.topic_model_class.__module__
            self.topic_model_class_string = self.topic_model_class.__name__
        kwargs['ignore'] = frozenset(kwargs.get('ignore', ())).union(('topic_model_class',))
        super(EnsembleLda, self).save(*args, **kwargs)
    save.__doc__ = SaveLoad.save.__doc__

    def convert_to_memory_friendly(self):
        if False:
            for i in range(10):
                print('nop')
        "Remove the stored gensim models and only keep their ttdas.\n\n        This frees up memory, but you won't have access to the individual  models anymore if you intended to use them\n        outside of the ensemble.\n        "
        self.tms = []
        self.memory_friendly_ttda = True

    def generate_gensim_representation(self):
        if False:
            return 10
        'Create a gensim model from the stable topics.\n\n        The returned representation is an Gensim LdaModel (:py:class:`gensim.models.LdaModel`) that has been\n        instantiated with an A-priori belief on word probability, eta, that represents the topic-term distributions of\n        any stable topics the were found by clustering over the ensemble of topic distributions.\n\n        When no stable topics have been detected, None is returned.\n\n        Returns\n        -------\n        :py:class:`gensim.models.LdaModel`\n            A Gensim LDA Model classic_model_representation for which:\n            ``classic_model_representation.get_topics() == self.get_topics()``\n\n        '
        logger.info('generating classic gensim model representation based on results from the ensemble')
        sstats_sum = self.sstats_sum
        if sstats_sum == 0 and 'corpus' in self.gensim_kw_args and (not self.gensim_kw_args['corpus'] is None):
            for document in self.gensim_kw_args['corpus']:
                for token in document:
                    sstats_sum += token[1]
            self.sstats_sum = sstats_sum
        stable_topics = self.get_topics()
        num_stable_topics = len(stable_topics)
        if num_stable_topics == 0:
            logger.error('the model did not detect any stable topic. You can try to adjust epsilon: recluster(eps=...)')
            self.classic_model_representation = None
            return
        params = self.gensim_kw_args.copy()
        params['eta'] = self.eta
        params['num_topics'] = num_stable_topics
        params['passes'] = 0
        classic_model_representation = self.get_topic_model_class()(**params)
        eta = classic_model_representation.eta
        if sstats_sum == 0:
            sstats_sum = classic_model_representation.state.sstats.sum()
            self.sstats_sum = sstats_sum
        eta_sum = 0
        if isinstance(eta, (int, float)):
            eta_sum = [eta * len(stable_topics[0])] * num_stable_topics
        else:
            if len(eta.shape) == 1:
                eta_sum = [[eta.sum()]] * num_stable_topics
            if len(eta.shape) > 1:
                eta_sum = np.array(eta.sum(axis=1)[:, None])
        normalization_factor = np.array([[sstats_sum / num_stable_topics]] * num_stable_topics) + eta_sum
        sstats = stable_topics * normalization_factor
        sstats -= eta
        classic_model_representation.state.sstats = sstats.astype(np.float32)
        classic_model_representation.sync_state()
        self.classic_model_representation = classic_model_representation
        return classic_model_representation

    def add_model(self, target, num_new_models=None):
        if False:
            print('Hello World!')
        'Add the topic term distribution array (ttda) of another model to the ensemble.\n\n        This way, multiple topic models can be connected to an ensemble manually. Make sure that all the models use\n        the exact same dictionary/idword mapping.\n\n        In order to generate new stable topics afterwards, use:\n            2. ``self.``:meth:`~gensim.models.ensemblelda.EnsembleLda.recluster`\n\n        The ttda of another ensemble can also be used, in that case set ``num_new_models`` to the ``num_models``\n        parameter of the ensemble, that means the number of classic models in the ensemble that generated the ttda.\n        This is important, because that information is used to estimate "min_samples" for _generate_topic_clusters.\n\n        If you trained this ensemble in the past with a certain Dictionary that you want to reuse for other\n        models, you can get it from: ``self.id2word``.\n\n        Parameters\n        ----------\n        target : {see description}\n            1. A single EnsembleLda object\n            2. List of EnsembleLda objects\n            3. A single Gensim topic model (e.g. (:py:class:`gensim.models.LdaModel`)\n            4. List of Gensim topic models\n\n            if memory_friendly_ttda is True, target can also be:\n            5. topic-term-distribution-array\n\n            example: [[0.1, 0.1, 0.8], [...], ...]\n\n            [topic1, topic2, ...]\n            with topic being an array of probabilities:\n            [token1, token2, ...]\n\n            token probabilities in a single topic sum to one, therefore, all the words sum to len(ttda)\n\n        num_new_models : integer, optional\n            the model keeps track of how many models were used in this ensemble. Set higher if ttda contained topics\n            from more than one model. Default: None, which takes care of it automatically.\n\n            If target is a 2D-array of float values, it assumes 1.\n\n            If the ensemble has ``memory_friendly_ttda`` set to False, then it will always use the number of models in\n            the target parameter.\n\n        '
        if not isinstance(target, (np.ndarray, list)):
            target = np.array([target])
        else:
            target = np.array(target)
            assert len(target) > 0
        if self.memory_friendly_ttda:
            detected_num_models = 0
            ttda = []
            if isinstance(target.dtype.type(), (np.number, float)):
                ttda = target
                detected_num_models = 1
            elif isinstance(target[0], type(self)):
                ttda = np.concatenate([ensemble.ttda for ensemble in target], axis=0)
                detected_num_models = sum([ensemble.num_models for ensemble in target])
            elif isinstance(target[0], basemodel.BaseTopicModel):
                ttda = np.concatenate([model.get_topics() for model in target], axis=0)
                detected_num_models = len(target)
            else:
                raise ValueError(f'target is of unknown type or a list of unknown types: {type(target[0])}')
            if num_new_models is None:
                self.num_models += detected_num_models
            else:
                self.num_models += num_new_models
        else:
            ttda = []
            if isinstance(target.dtype.type(), (np.number, float)):
                raise ValueError('ttda arrays cannot be added to ensembles, for which memory_friendly_ttda=False, you can call convert_to_memory_friendly, but it will discard the stored gensim models and only keep the relevant topic term distributions from them.')
            elif isinstance(target[0], type(self)):
                for ensemble in target:
                    self.tms += ensemble.tms
                ttda = np.concatenate([ensemble.ttda for ensemble in target], axis=0)
            elif isinstance(target[0], basemodel.BaseTopicModel):
                self.tms += target.tolist()
                ttda = np.concatenate([model.get_topics() for model in target], axis=0)
            else:
                raise ValueError(f'target is of unknown type or a list of unknown types: {type(target[0])}')
            if num_new_models is not None and num_new_models + self.num_models != len(self.tms):
                logger.info('num_new_models will be ignored. num_models should match the number of stored models for a memory unfriendly ensemble')
            self.num_models = len(self.tms)
        logger.info(f'ensemble contains {self.num_models} models and {len(self.ttda)} topics now')
        if self.ttda.shape[1] != ttda.shape[1]:
            raise ValueError(f'target ttda dimensions do not match. Topics must be {self.ttda.shape[-1]} but was {ttda.shape[-1]} elements large')
        self.ttda = np.append(self.ttda, ttda, axis=0)
        self.asymmetric_distance_matrix_outdated = True

    def _generate_asymmetric_distance_matrix(self):
        if False:
            print('Hello World!')
        'Calculate the pairwise distance matrix for all the ttdas from the ensemble.\n\n        Returns the asymmetric pairwise distance matrix that is used in the DBSCAN clustering.\n\n        Afterwards, the model needs to be reclustered for this generated matrix to take effect.\n\n        '
        workers = self.distance_workers
        self.asymmetric_distance_matrix_outdated = False
        logger.info(f'generating a {len(self.ttda)} x {len(self.ttda)} asymmetric distance matrix...')
        if workers is not None and workers <= 1:
            self.asymmetric_distance_matrix = _calculate_asymmetric_distance_matrix_chunk(ttda1=self.ttda, ttda2=self.ttda, start_index=0, masking_method=self.masking_method, masking_threshold=self.masking_threshold)
        else:
            if workers is None:
                workers = os.cpu_count()
            self.asymmetric_distance_matrix = _calculate_assymetric_distance_matrix_multiproc(workers=workers, entire_ttda=self.ttda, masking_method=self.masking_method, masking_threshold=self.masking_threshold)

    def _generate_topic_clusters(self, eps=0.1, min_samples=None):
        if False:
            i = 10
            return i + 15
        'Run the CBDBSCAN algorithm on all the detected topics and label them with label-indices.\n\n        The final approval and generation of stable topics is done in ``_generate_stable_topics()``.\n\n        Parameters\n        ----------\n        eps : float\n            dbscan distance scale\n        min_samples : int, optional\n            defaults to ``int(self.num_models / 2)``, dbscan min neighbours threshold required to consider\n            a topic to be a core. Should scale with the number of models, ``self.num_models``\n\n        '
        if min_samples is None:
            min_samples = int(self.num_models / 2)
            logger.info('fitting the clustering model, using %s for min_samples', min_samples)
        else:
            logger.info('fitting the clustering model')
        self.cluster_model = CBDBSCAN(eps=eps, min_samples=min_samples)
        self.cluster_model.fit(self.asymmetric_distance_matrix)

    def _generate_stable_topics(self, min_cores=None):
        if False:
            i = 10
            return i + 15
        'Generate stable topics out of the clusters.\n\n        The function finds clusters of topics using a variant of DBScan.  If a cluster has enough core topics\n        (c.f. parameter ``min_cores``), then this cluster represents a stable topic.  The stable topic is specifically\n        calculated as the average over all topic-term distributions of the core topics in the cluster.\n\n        This function is the last step that has to be done in the ensemble.  After this step is complete,\n        Stable topics can be retrieved afterwards using the :meth:`~gensim.models.ensemblelda.EnsembleLda.get_topics`\n        method.\n\n        Parameters\n        ----------\n        min_cores : int\n            Minimum number of core topics needed to form a cluster that represents a stable topic.\n                Using ``None`` defaults to ``min_cores = min(3, max(1, int(self.num_models /4 +1)))``\n\n        '
        if min_cores == 0:
            min_cores = 1
        if min_cores is None:
            min_cores = min(3, max(1, int(self.num_models / 4 + 1)))
            logger.info('generating stable topics, using %s for min_cores', min_cores)
        else:
            logger.info('generating stable topics')
        cbdbscan_topics = self.cluster_model.results
        grouped_by_labels = _group_by_labels(cbdbscan_topics)
        clusters = _aggregate_topics(grouped_by_labels)
        valid_clusters = _validate_clusters(clusters, min_cores)
        valid_cluster_labels = {cluster.label for cluster in valid_clusters}
        for topic in cbdbscan_topics:
            topic.valid_neighboring_labels = {label for label in topic.neighboring_labels if label in valid_cluster_labels}
        valid_core_mask = np.vectorize(_is_valid_core)(cbdbscan_topics)
        valid_topics = self.ttda[valid_core_mask]
        topic_labels = np.array([topic.label for topic in cbdbscan_topics])[valid_core_mask]
        unique_labels = np.unique(topic_labels)
        num_stable_topics = len(unique_labels)
        stable_topics = np.empty((num_stable_topics, len(self.id2word)))
        for (label_index, label) in enumerate(unique_labels):
            topics_of_cluster = np.array([topic for (t, topic) in enumerate(valid_topics) if topic_labels[t] == label])
            stable_topics[label_index] = topics_of_cluster.mean(axis=0)
        self.valid_clusters = valid_clusters
        self.stable_topics = stable_topics
        logger.info('found %s stable topics', len(stable_topics))

    def recluster(self, eps=0.1, min_samples=None, min_cores=None):
        if False:
            return 10
        'Reapply CBDBSCAN clustering and stable topic generation.\n\n        Stable topics can be retrieved using :meth:`~gensim.models.ensemblelda.EnsembleLda.get_topics`.\n\n        Parameters\n        ----------\n        eps : float\n            epsilon for the CBDBSCAN algorithm, having the same meaning as in classic DBSCAN clustering.\n            default: ``0.1``\n        min_samples : int\n            The minimum number of samples in the neighborhood of a topic to be considered a core in CBDBSCAN.\n            default: ``int(self.num_models / 2)``\n        min_cores : int\n            how many cores a cluster has to have, to be treated as stable topic. That means, how many topics\n            that look similar have to be present, so that the average topic in those is used as stable topic.\n            default: ``min(3, max(1, int(self.num_models /4 +1)))``\n\n        '
        if self.asymmetric_distance_matrix_outdated:
            logger.info('asymmetric distance matrix is outdated due to add_model')
            self._generate_asymmetric_distance_matrix()
        self._generate_topic_clusters(eps, min_samples)
        self._generate_stable_topics(min_cores)
        self.generate_gensim_representation()

    def get_topics(self):
        if False:
            for i in range(10):
                print('nop')
        'Return only the stable topics from the ensemble.\n\n        Returns\n        -------\n        2D Numpy.numpy.ndarray of floats\n            List of stable topic term distributions\n\n        '
        return self.stable_topics

    def _ensure_gensim_representation(self):
        if False:
            i = 10
            return i + 15
        'Check if stable topics and the internal gensim representation exist. Raise an error if not.'
        if self.classic_model_representation is None:
            if len(self.stable_topics) == 0:
                raise ValueError('no stable topic was detected')
            else:
                raise ValueError('use generate_gensim_representation() first')

    def __getitem__(self, i):
        if False:
            return 10
        'See :meth:`gensim.models.LdaModel.__getitem__`.'
        self._ensure_gensim_representation()
        return self.classic_model_representation[i]

    def inference(self, *posargs, **kwargs):
        if False:
            return 10
        'See :meth:`gensim.models.LdaModel.inference`.'
        self._ensure_gensim_representation()
        return self.classic_model_representation.inference(*posargs, **kwargs)

    def log_perplexity(self, *posargs, **kwargs):
        if False:
            i = 10
            return i + 15
        'See :meth:`gensim.models.LdaModel.log_perplexity`.'
        self._ensure_gensim_representation()
        return self.classic_model_representation.log_perplexity(*posargs, **kwargs)

    def print_topics(self, *posargs, **kwargs):
        if False:
            print('Hello World!')
        'See :meth:`gensim.models.LdaModel.print_topics`.'
        self._ensure_gensim_representation()
        return self.classic_model_representation.print_topics(*posargs, **kwargs)

    @property
    def id2word(self):
        if False:
            while True:
                i = 10
        'Return the :py:class:`gensim.corpora.dictionary.Dictionary` object used in the model.'
        return self.gensim_kw_args['id2word']

class CBDBSCAN:
    """A Variation of the DBSCAN algorithm called Checkback DBSCAN (CBDBSCAN).

    The algorithm works based on DBSCAN-like parameters 'eps' and 'min_samples' that respectively define how far a
    "nearby" point is, and the minimum number of nearby points needed to label a candidate datapoint a core of a
    cluster. (See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).

    The algorithm works as follows:

    1. (A)symmetric distance matrix provided at fit-time (called 'amatrix').
       For the sake of example below, assume the there are only five topics (amatrix contains distances with dim 5x5),
       T_1, T_2, T_3, T_4, T_5:
    2. Start by scanning a candidate topic with respect to a parent topic
       (e.g. T_1 with respect to parent None)
    3. Check which topics are nearby the candidate topic using 'self.eps' as a threshold and call them neighbours
       (e.g. assume T_3, T_4, and T_5 are nearby and become neighbours)
    4. If there are more neighbours than 'self.min_samples', the candidate topic becomes a core candidate for a cluster
       (e.g. if 'min_samples'=1, then T_1 becomes the first core of a cluster)
    5. If candidate is a core, CheckBack (CB) to find the fraction of neighbours that are either the parent or the
       parent's neighbours.  If this fraction is more than 75%, give the candidate the same label as its parent.
       (e.g. in the trivial case there is no parent (or neighbours of that parent), a new incremental label is given)
    6. If candidate is a core, recursively scan the next nearby topic (e.g. scan T_3) labeling the previous topic as
       the parent and the previous neighbours as the parent_neighbours - repeat steps 2-6:

       2. (e.g. Scan candidate T_3 with respect to parent T_1 that has parent_neighbours T_3, T_4, and T_5)
       3. (e.g. T5 is the only neighbour)
       4. (e.g. number of neighbours is 1, therefore candidate T_3 becomes a core)
       5. (e.g. CheckBack finds that two of the four parent and parent neighbours are neighbours of candidate T_3.
          Therefore the candidate T_3 does NOT get the same label as its parent T_1)
       6. (e.g. Scan candidate T_5 with respect to parent T_3 that has parent_neighbours T_5)

    The CB step has the effect that it enforces cluster compactness and allows the model to avoid creating clusters for
    unstable topics made of a composition of multiple stable topics.

    """

    def __init__(self, eps, min_samples):
        if False:
            return 10
        'Create a new CBDBSCAN object. Call fit in order to train it on an asymmetric distance matrix.\n\n        Parameters\n        ----------\n        eps : float\n            epsilon for the CBDBSCAN algorithm, having the same meaning as in classic DBSCAN clustering.\n        min_samples : int\n            The minimum number of samples in the neighborhood of a topic to be considered a core in CBDBSCAN.\n\n        '
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, amatrix):
        if False:
            while True:
                i = 10
        'Apply the algorithm to an asymmetric distance matrix.'
        self.next_label = 0
        topic_clustering_results = [Topic(is_core=False, neighboring_labels=set(), neighboring_topic_indices=set(), label=None, num_neighboring_labels=0, valid_neighboring_labels=set()) for i in range(len(amatrix))]
        amatrix_copy = amatrix.copy()
        np.fill_diagonal(amatrix_copy, 1)
        min_distance_per_topic = [(distance, index) for (index, distance) in enumerate(amatrix_copy.min(axis=1))]
        min_distance_per_topic_sorted = sorted(min_distance_per_topic, key=lambda distance: distance[0])
        ordered_min_similarity = [index for (distance, index) in min_distance_per_topic_sorted]

        def scan_topic(topic_index, current_label=None, parent_neighbors=None):
            if False:
                i = 10
                return i + 15
            'Extend the cluster in one direction.\n\n            Results are accumulated to ``self.results``.\n\n            Parameters\n            ----------\n            topic_index : int\n                The topic that might be added to the existing cluster, or which might create a new cluster if necessary.\n            current_label : int\n                The label of the cluster that might be suitable for ``topic_index``\n\n            '
            neighbors_sorted = sorted([(distance, index) for (index, distance) in enumerate(amatrix_copy[topic_index])], key=lambda x: x[0])
            neighboring_topic_indices = [index for (distance, index) in neighbors_sorted if distance < self.eps]
            num_neighboring_topics = len(neighboring_topic_indices)
            if num_neighboring_topics >= self.min_samples:
                topic_clustering_results[topic_index].is_core = True
                if current_label is None:
                    current_label = self.next_label
                    self.next_label += 1
                else:
                    close_parent_neighbors_mask = amatrix_copy[topic_index][parent_neighbors] < self.eps
                    if close_parent_neighbors_mask.mean() < 0.25:
                        current_label = self.next_label
                        self.next_label += 1
                topic_clustering_results[topic_index].label = current_label
                for neighboring_topic_index in neighboring_topic_indices:
                    if topic_clustering_results[neighboring_topic_index].label is None:
                        ordered_min_similarity.remove(neighboring_topic_index)
                        scan_topic(neighboring_topic_index, current_label, neighboring_topic_indices + [topic_index])
                    topic_clustering_results[neighboring_topic_index].neighboring_topic_indices.add(topic_index)
                    topic_clustering_results[neighboring_topic_index].neighboring_labels.add(current_label)
            elif current_label is None:
                topic_clustering_results[topic_index].label = -1
            else:
                topic_clustering_results[topic_index].label = current_label
        while len(ordered_min_similarity) != 0:
            next_topic_index = ordered_min_similarity.pop(0)
            scan_topic(next_topic_index)
        self.results = topic_clustering_results