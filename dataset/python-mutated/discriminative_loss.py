from chainer import backend
from chainer.functions.activation.relu import relu
from chainer.functions.array.broadcast import broadcast_to
from chainer.functions.math.basic_math import absolute
from chainer.functions.math.sqrt import sqrt
from chainer.functions.math.sum import sum as c_sum

class DiscriminativeMarginBasedClusteringLoss(object):
    """Discriminative margin-based clustering loss function

    This is the implementation of the following paper:
    https://arxiv.org/abs/1708.02551
    This method is a semi-supervised solution to instance segmentation.
    It calculates pixel embeddings, and calculates three different terms
    based on those embeddings and applies them as loss.
    The main idea is that the pixel embeddings
    for same instances have to be closer to each other (pull force),
    for different instances, they have to be further away (push force).
    The loss also brings a weak regularization term to prevent overfitting.
    This loss function calculates the following three parameters:

    Variance Loss
        Loss to penalize distances between pixels which are belonging
        to the same instance. (Pull force)

    Distance loss
        Loss to penalize distances between the centers of instances.
        (Push force)

    Regularization loss
        Small regularization loss to penalize weights against overfitting.

    """

    def __init__(self, delta_v=0.5, delta_d=1.5, max_embedding_dim=10, norm=1, alpha=1.0, beta=1.0, gamma=0.001):
        if False:
            i = 10
            return i + 15
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_embedding_dim = max_embedding_dim
        if self.max_embedding_dim <= 0:
            raise ValueError('Max number of embeddings has to be positive!')
        if norm == 1:
            self.norm = lambda x, axis=None: c_sum(absolute(x), axis=axis)
        elif norm == 2:
            self.norm = lambda x, axis=None: sqrt(c_sum(x ** 2, axis=axis))
        else:
            raise ValueError('For discriminative loss, norm can only be 1 or 2. Obtained the value : {}'.format(norm))

    def __call__(self, embeddings, labels):
        if False:
            print('Hello World!')
        '\n        Args:\n            embeddings (:class:`~chainer.Variable` or :ref:`ndarray`):\n                predicted embedding vectors\n                (batch size, max embedding dimensions, height, width)\n\n            labels (:ref:`ndarray`):\n                instance segmentation ground truth\n                each unique value has to be denoting one instance\n                (batch size, height, width)\n\n        Returns:\n            :class:`tuple` of :class:`chainer.Variable`:\n            - *Variance loss*: Variance loss multiplied by ``alpha``\n            - *Distance loss*: Distance loss multiplied by ``beta``\n            - *Regularization loss*: Regularization loss multiplied by\n              ``gamma``\n\n        '
        assert self.max_embedding_dim == embeddings.shape[1]
        l_dist = 0.0
        count = 0
        xp = backend.get_array_module(embeddings)
        emb = embeddings[None, :]
        emb = broadcast_to(emb, (emb.shape[1], emb.shape[1], emb.shape[2], emb.shape[3], emb.shape[4]))
        ms = []
        for c in range(self.max_embedding_dim):
            mask = xp.expand_dims(labels == c + 1, 1)
            ms.append(mask)
        if hasattr(xp, 'stack'):
            ms = xp.stack(ms, 0)
        else:
            ms = xp.concatenate([xp.expand_dims(x, 0) for x in ms], 0)
        mns = c_sum(emb * ms, axis=(3, 4))
        mns = mns / xp.maximum(xp.sum(ms, (2, 3, 4))[:, :, None], 1)
        mns_exp = mns[:, :, :, None, None]
        l_reg = c_sum(self.norm(mns, (1, 2)))
        l_reg = l_reg / (self.max_embedding_dim * embeddings.shape[0])
        l_var = self.norm((mns_exp - emb) * ms, 2)
        l_var = relu(l_var - self.delta_v) ** 2
        l_var = c_sum(l_var, (1, 2, 3))
        l_var = l_var / xp.maximum(xp.sum(ms, (1, 2, 3, 4)), 1)
        l_var = c_sum(l_var) / self.max_embedding_dim
        for c_a in range(len(mns)):
            for c_b in range(c_a + 1, len(mns)):
                m_a = mns[c_a]
                m_b = mns[c_b]
                dist = self.norm(m_a - m_b, 1)
                l_dist += c_sum(relu(2 * self.delta_d - dist) ** 2)
                count += 1
        l_dist /= max(count * embeddings.shape[0], 1)
        rtn = (self.alpha * l_var, self.beta * l_dist, self.gamma * l_reg)
        return rtn

def discriminative_margin_based_clustering_loss(embeddings, labels, delta_v, delta_d, max_embedding_dim, norm=1, alpha=1.0, beta=1.0, gamma=0.001):
    if False:
        print('Hello World!')
    'Discriminative margin-based clustering loss function\n\n    This is the implementation of the following paper:\n    https://arxiv.org/abs/1708.02551\n    This method is a semi-supervised solution to instance segmentation.\n    It calculates pixel embeddings, and calculates three different terms\n    based on those embeddings and applies them as loss.\n    The main idea is that the pixel embeddings\n    for same instances have to be closer to each other (pull force),\n    for different instances, they have to be further away (push force).\n    The loss also brings a weak regularization term to prevent overfitting.\n    This loss function calculates the following three parameters:\n\n    Variance Loss\n        Loss to penalize distances between pixels which are belonging\n        to the same instance. (Pull force)\n\n    Distance loss\n        Loss to penalize distances between the centers of instances.\n        (Push force)\n\n    Regularization loss\n        Small regularization loss to penalize weights against overfitting.\n\n    Args:\n        embeddings (:class:`~chainer.Variable` or :ref:`ndarray`):\n            predicted embedding vectors\n            (batch size, max embedding dimensions, height, width)\n\n        labels (:ref:`ndarray`):\n            instance segmentation ground truth\n            each unique value has to be denoting one instance\n            (batch size, height, width)\n        delta_v (float): Minimum distance to start penalizing variance\n        delta_d (float): Maximum distance to stop penalizing distance\n        max_embedding_dim (int): Maximum number of embedding dimensions\n        norm (int): Norm to calculate pixels and cluster center distances\n        alpha (float): Weight for variance loss\n        beta (float): Weight for distance loss\n        gamma (float): Weight for regularization loss\n\n    Returns:\n        :class:`tuple` of :class:`chainer.Variable`:\n        - *Variance loss*: Variance loss multiplied by ``alpha``\n        - *Distance loss*: Distance loss multiplied by ``beta``\n        - *Regularization loss*: Regularization loss multiplied by ``gamma``\n\n    '
    loss = DiscriminativeMarginBasedClusteringLoss(delta_v, delta_d, max_embedding_dim, norm, alpha, beta, gamma)
    return loss(embeddings, labels)