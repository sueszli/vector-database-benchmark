import numpy
from annoy import AnnoyIndex

def test_seeding():
    if False:
        return 10
    f = 10
    X = numpy.random.rand(1000, f)
    Y = numpy.random.rand(50, f)
    indexes = []
    for i in range(2):
        index = AnnoyIndex(f, 'angular')
        index.set_seed(42)
        for j in range(X.shape[0]):
            index.add_item(j, X[j])
        index.build(10)
        indexes.append(index)
    for k in range(Y.shape[0]):
        assert indexes[0].get_nns_by_vector(Y[k], 100) == indexes[1].get_nns_by_vector(Y[k], 100)