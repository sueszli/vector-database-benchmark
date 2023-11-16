import numpy as np
import scipy.spatial
from Orange.preprocess.preprocess import RemoveNaNRows, Continuize, Scale
from Orange.projection import LinearProjector, DomainProjection
__all__ = ['FreeViz']

class FreeVizModel(DomainProjection):
    var_prefix = 'freeviz'

class FreeViz(LinearProjector):
    name = 'FreeViz'
    supports_sparse = False
    preprocessors = [RemoveNaNRows(), Continuize(multinomial_treatment=Continuize.FirstAsBase), Scale(scale=Scale.Span)]
    projection = FreeVizModel

    def __init__(self, weights=None, center=True, scale=True, dim=2, p=1, initial=None, maxiter=500, alpha=0.1, gravity=None, atol=1e-05, preprocessors=None):
        if False:
            while True:
                i = 10
        super().__init__(preprocessors=preprocessors)
        self.weights = weights
        self.center = center
        self.scale = scale
        self.dim = dim
        self.p = p
        self.initial = initial
        self.maxiter = maxiter
        self.alpha = alpha
        self.atol = atol
        self.gravity = gravity
        self.is_class_discrete = False
        self.components_ = None

    def __call__(self, data):
        if False:
            for i in range(10):
                print('nop')
        if data is not None:
            self.is_class_discrete = data.domain.class_var.is_discrete
            if len([attr for attr in data.domain.attributes if attr.is_discrete and len(attr.values) > 2]):
                raise ValueError('Can not handle discrete variables with more than two values')
        return super().__call__(data)

    def get_components(self, X, Y):
        if False:
            for i in range(10):
                print('nop')
        return self.freeviz(X, Y, weights=self.weights, center=self.center, scale=self.scale, dim=self.dim, p=self.p, initial=self.initial, maxiter=self.maxiter, alpha=self.alpha, atol=self.atol, gravity=self.gravity, is_class_discrete=self.is_class_discrete)[1].T

    @classmethod
    def squareform(cls, d):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        d : (N * (N - 1) // 2, ) ndarray\n            A hollow symmetric square array in condensed form\n\n        Returns\n        -------\n        D : (N, N) ndarray\n            A symmetric square array in redundant form.\n\n        See also\n        --------\n        scipy.spatial.distance.squareform\n        '
        assert d.ndim == 1
        return scipy.spatial.distance.squareform(d, checks=False)

    @classmethod
    def row_v(cls, a):
        if False:
            print('Hello World!')
        '\n        Return a view of `a` as a row vector.\n        '
        return a.reshape((1, -1))

    @classmethod
    def col_v(cls, a):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a view of `a` as a column vector.\n        '
        return a.reshape((-1, 1))

    @classmethod
    def allclose(cls, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        if False:
            for i in range(10):
                print('nop')
        return np.all(np.isclose(a, b, rtol, atol, equal_nan=equal_nan))

    @classmethod
    def forces_regression(cls, distances, y, p=1):
        if False:
            return 10
        y = np.asarray(y)
        ydist = scipy.spatial.distance.pdist(y.reshape(-1, 1), 'sqeuclidean')
        mask = distances > np.finfo(distances.dtype).eps * 100
        F = ydist
        if p == 1:
            F[mask] /= distances[mask]
        else:
            F[mask] /= distances[mask] ** p
        return F

    @classmethod
    def forces_classification(cls, distances, y, p=1, gravity=None):
        if False:
            print('Hello World!')
        diffclass = scipy.spatial.distance.pdist(y.reshape(-1, 1), 'hamming') != 0
        if p == 1:
            F = -distances
        else:
            F = -distances ** p
        mask = diffclass & (distances > np.finfo(distances.dtype).eps * 100)
        assert mask.shape == F.shape and mask.dtype == bool
        if p == 1:
            F[mask] = 1 / distances[mask]
        else:
            F[mask] = 1 / distances[mask] ** p
        if gravity is not None:
            F[mask] *= -np.sum(F[~mask]) / np.sum(F[mask]) / gravity
        return F

    @classmethod
    def gradient(cls, X, embeddings, forces, embedding_dist=None, weights=None):
        if False:
            for i in range(10):
                print('nop')
        X = np.asarray(X)
        embeddings = np.asarray(embeddings)
        if weights is not None:
            weights = np.asarray(weights)
            if weights.ndim != 1:
                raise ValueError('weights.ndim != 1 ({})'.format(weights.ndim))
        (N, P) = X.shape
        (_, dim) = embeddings.shape
        if not N == embeddings.shape[0]:
            raise ValueError('X and embeddings must have the same length ({}!={})'.format(X.shape[0], embeddings.shape[0]))
        if weights is not None and X.shape[0] != weights.shape[0]:
            raise ValueError('X.shape[0] != weights.shape[0] ({}!={})'.format(X.shape[0], weights.shape[0]))
        embedding_diff = embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :]
        assert embedding_diff.shape == (N, N, dim)
        assert cls.allclose(embedding_diff[0, 1], embeddings[0] - embeddings[1])
        assert cls.allclose(embedding_diff[1, 0], -embedding_diff[0, 1])
        if embedding_dist is not None:
            diff_norm = cls.squareform(embedding_dist)
        else:
            diff_norm = np.linalg.norm(embedding_diff, axis=2)
        mask = diff_norm > np.finfo(diff_norm.dtype).eps * 100
        embedding_diff[mask] /= diff_norm[mask][:, np.newaxis]
        forces = cls.squareform(forces)
        if weights is not None:
            forces *= cls.row_v(weights)
            forces *= cls.col_v(weights)
        F = embedding_diff * forces[:, :, np.newaxis]
        assert F.shape == (N, N, dim)
        F = np.sum(F, axis=0)
        assert F.shape == (N, dim)
        G = X.T.dot(F)
        assert G.shape == (P, dim)
        return G

    @classmethod
    def freeviz_gradient(cls, X, y, embedding, p=1, weights=None, gravity=None, is_class_discrete=False):
        if False:
            i = 10
            return i + 15
        "\n        Return the gradient for the FreeViz [1]_ projection.\n\n        Parameters\n        ----------\n        X : (N, P) ndarray\n            The data instance coordinates\n        y : (N,) ndarray\n            The instance target/class values\n        embedding : (N, dim) ndarray\n            The current FreeViz point embeddings.\n        p : positive number\n            The force 'power', e.g. if p=1 (default) the attractive/repulsive\n            forces follow linear/inverse linear law, for p=2 the forces follow\n            square/inverse square law, ...\n        weights : (N, ) ndarray, optional\n            Optional vector of sample weights.\n\n        Returns\n        -------\n        G : (P, dim) ndarray\n            The projection gradient.\n\n        .. [1] Janez Demsar, Gregor Leban, Blaz Zupan\n               FreeViz - An Intelligent Visualization Approach for Class-Labeled\n               Multidimensional Data Sets, Proceedings of IDAMAP 2005, Edinburgh.\n        "
        X = np.asarray(X)
        y = np.asarray(y)
        embedding = np.asarray(embedding)
        assert X.ndim == 2 and X.shape[0] == y.shape[0] == embedding.shape[0]
        D = scipy.spatial.distance.pdist(embedding)
        if is_class_discrete:
            forces = cls.forces_classification(D, y, p=p, gravity=gravity)
        else:
            forces = cls.forces_regression(D, y, p=p)
        G = cls.gradient(X, embedding, forces, embedding_dist=D, weights=weights)
        return G

    @classmethod
    def _rotate(cls, A):
        if False:
            print('Hello World!')
        '\n        Rotate a 2D projection A so the first axis (row in A) is aligned with\n        vector (1, 0).\n        '
        assert A.ndim == 2 and A.shape[1] == 2
        phi = np.arctan2(A[0, 1], A[0, 0])
        R = [[np.cos(-phi), np.sin(-phi)], [-np.sin(-phi), np.cos(-phi)]]
        return np.dot(A, R)

    @classmethod
    def freeviz(cls, X, y, weights=None, center=True, scale=True, dim=2, p=1, initial=None, maxiter=500, alpha=0.1, atol=1e-05, gravity=None, is_class_discrete=False):
        if False:
            while True:
                i = 10
        "\n        FreeViz\n\n        Compute a linear lower dimensional projection to optimize separation\n        between classes ([1]_).\n\n        Parameters\n        ----------\n        X : (N, P) ndarray\n            The input data instances\n        y : (N, ) ndarray\n            The instance class labels\n        weights : (N, ) ndarray, optional\n            Instance weights\n        center : bool or (P,) ndarray\n            If `True` then X will have mean subtracted out, if False no\n            centering is performed. Alternatively can be a P vector to subtract\n            from X.\n        scale : bool or (P,) ndarray\n            If `True` the X's column will be scaled by 1/SD, if False no scaling\n            is performed. Alternatively can be a P vector to divide X by.\n        dim : int\n            The dimension of the projected points/embedding.\n        p : positive number\n            The force 'power', e.g. if p=1 (default) the attractive/repulsive\n            forces follow linear/inverse linear law, for p=2 the forces follow\n            square/inverse square law, ...\n        initial : (P, dim) ndarray, optional\n            Initial projection matrix\n        maxiter : int\n            Maximum number of iterations.\n        alpha : float\n            The step size ('learning rate')\n        atol : float\n            Terminating numerical tolerance (absolute).\n\n        Returns\n        -------\n        embeddings : (N, dim) ndarray\n            The point projections (`= X.dot(P)`)\n        projection : (P, dim)\n            The projection matrix.\n        center : (P,) ndarray or None\n            The translation applied to X (if any).\n        scale : (P,) ndarray or None\n            The scaling applied to X (if any).\n\n        .. [1] Janez Demsar, Gregor Leban, Blaz Zupan\n               FreeViz - An Intelligent Visualization Approach for Class-Labeled\n               Multidimensional Data Sets, Proceedings of IDAMAP 2005, Edinburgh.\n        "
        needcopy = center is not False or scale is not False
        X = np.array(X, copy=needcopy)
        y = np.asarray(y)
        (N, P) = X.shape
        (_N,) = y.shape
        if N != _N:
            raise ValueError('X and y must have the same length')
        if weights is not None:
            weights = np.asarray(weights)
        if isinstance(center, bool):
            if center:
                center = np.mean(X, axis=0)
            else:
                center = None
        else:
            center = np.asarray(center, dtype=X.dtype)
            if center.shape != (P,):
                raise ValueError('center.shape != (X.shape[1], ) ({} != {})'.format(center.shape, (X.shape[1],)))
        if isinstance(scale, bool):
            if scale:
                scale = np.std(X, axis=0)
            else:
                scale = None
        else:
            scale = np.asarray(scale, dtype=X.dtype)
            if scale.shape != (P,):
                raise ValueError('scale.shape != (X.shape[1],) ({} != {}))'.format(scale.shape, (P,)))
        if initial is not None:
            initial = np.asarray(initial)
            if initial.ndim != 2 or initial.shape != (P, dim):
                raise ValueError
        else:
            initial = cls.init_random(P, dim)
        if center is not None:
            X -= center
        if scale is not None:
            scalenonzero = np.abs(scale) > np.finfo(scale.dtype).eps
            X[:, scalenonzero] /= scale[scalenonzero]
        A = initial
        embeddings = np.dot(X, A)
        step_i = 0
        while step_i < maxiter:
            G = cls.freeviz_gradient(X, y, embeddings, p=p, weights=weights, gravity=gravity, is_class_discrete=is_class_discrete)
            with np.errstate(divide='ignore'):
                step = np.min(np.linalg.norm(A, axis=1) / np.linalg.norm(G, axis=1))
                if not np.isfinite(step):
                    break
            step = alpha * step
            Anew = A - step * G
            Anew = Anew - np.mean(Anew, axis=0)
            maxr = np.max(np.linalg.norm(Anew, axis=1))
            if maxr >= 0.001:
                Anew /= maxr
            change = np.linalg.norm(Anew - A, axis=1)
            if cls.allclose(change, 0, atol=atol):
                break
            A = Anew
            embeddings = np.dot(X, A)
            step_i = step_i + 1
        if dim == 2:
            A = cls._rotate(A)
        return (embeddings, A, center, scale)

    @staticmethod
    def init_radial(p):
        if False:
            print('Hello World!')
        '\n        Return a 2D projection with a circular anchor placement.\n        '
        assert p > 0
        if p == 1:
            axes_angle = [0]
        elif p == 2:
            axes_angle = [0, np.pi / 2]
        else:
            axes_angle = np.linspace(0, 2 * np.pi, p, endpoint=False)
        A = np.c_[np.cos(axes_angle), np.sin(axes_angle)]
        return A

    @staticmethod
    def init_random(p, dim, rstate=None):
        if False:
            print('Hello World!')
        if not isinstance(rstate, np.random.RandomState):
            rstate = np.random.RandomState(rstate if rstate is not None else 0)
        return rstate.rand(p, dim) * 2 - 1