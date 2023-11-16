import logging
from gensim import interfaces, matutils
logger = logging.getLogger(__name__)

class NormModel(interfaces.TransformationABC):
    """Objects of this class realize the explicit normalization of vectors (l1 and l2)."""

    def __init__(self, corpus=None, norm='l2'):
        if False:
            i = 10
            return i + 15
        "Compute the l1 or l2 normalization by normalizing separately for each document in a corpus.\n\n        If :math:`v_{i,j}` is the 'i'th component of the vector representing document 'j', the l1 normalization is\n\n        .. math:: l1_{i, j} = \\frac{v_{i,j}}{\\sum_k |v_{k,j}|}\n\n        the l2 normalization is\n\n        .. math:: l2_{i, j} = \\frac{v_{i,j}}{\\sqrt{\\sum_k v_{k,j}^2}}\n\n\n        Parameters\n        ----------\n        corpus : iterable of iterable of (int, number), optional\n            Input corpus.\n        norm : {'l1', 'l2'}, optional\n            Norm used to normalize.\n\n        "
        self.norm = norm
        if corpus is not None:
            self.calc_norm(corpus)
        else:
            pass

    def __str__(self):
        if False:
            while True:
                i = 10
        return '%s<num_docs=%s, num_nnz=%s, norm=%s>' % (self.__class__.__name__, self.num_docs, self.num_nnz, self.norm)

    def calc_norm(self, corpus):
        if False:
            return 10
        'Calculate the norm by calling :func:`~gensim.matutils.unitvec` with the norm parameter.\n\n        Parameters\n        ----------\n        corpus : iterable of iterable of (int, number)\n            Input corpus.\n\n        '
        logger.info('Performing %s normalization...', self.norm)
        norms = []
        numnnz = 0
        docno = 0
        for bow in corpus:
            docno += 1
            numnnz += len(bow)
            norms.append(matutils.unitvec(bow, self.norm))
        self.num_docs = docno
        self.num_nnz = numnnz
        self.norms = norms

    def normalize(self, bow):
        if False:
            i = 10
            return i + 15
        'Normalize a simple count representation.\n\n        Parameters\n        ----------\n        bow : list of (int, number)\n            Document in BoW format.\n\n        Returns\n        -------\n        list of (int, number)\n            Normalized document.\n\n\n        '
        vector = matutils.unitvec(bow, self.norm)
        return vector

    def __getitem__(self, bow):
        if False:
            i = 10
            return i + 15
        'Call the :func:`~gensim.models.normmodel.NormModel.normalize`.\n\n        Parameters\n        ----------\n        bow : list of (int, number)\n            Document in BoW format.\n\n        Returns\n        -------\n        list of (int, number)\n            Normalized document.\n\n        '
        return self.normalize(bow)