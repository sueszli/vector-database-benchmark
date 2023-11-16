from paddle.base import core
__all__ = []

class Index:

    def __init__(self, name):
        if False:
            return 10
        self._name = name

class TreeIndex(Index):

    def __init__(self, name, path):
        if False:
            return 10
        super().__init__(name)
        self._wrapper = core.IndexWrapper()
        self._wrapper.insert_tree_index(name, path)
        self._tree = self._wrapper.get_tree_index(name)
        self._height = self._tree.height()
        self._branch = self._tree.branch()
        self._total_node_nums = self._tree.total_node_nums()
        self._emb_size = self._tree.emb_size()
        self._layerwise_sampler = None

    def height(self):
        if False:
            print('Hello World!')
        return self._height

    def branch(self):
        if False:
            for i in range(10):
                print('nop')
        return self._branch

    def total_node_nums(self):
        if False:
            i = 10
            return i + 15
        return self._total_node_nums

    def emb_size(self):
        if False:
            i = 10
            return i + 15
        return self._emb_size

    def get_all_leafs(self):
        if False:
            while True:
                i = 10
        return self._tree.get_all_leafs()

    def get_nodes(self, codes):
        if False:
            print('Hello World!')
        return self._tree.get_nodes(codes)

    def get_layer_codes(self, level):
        if False:
            print('Hello World!')
        return self._tree.get_layer_codes(level)

    def get_travel_codes(self, id, start_level=0):
        if False:
            while True:
                i = 10
        return self._tree.get_travel_codes(id, start_level)

    def get_ancestor_codes(self, ids, level):
        if False:
            i = 10
            return i + 15
        return self._tree.get_ancestor_codes(ids, level)

    def get_children_codes(self, ancestor, level):
        if False:
            for i in range(10):
                print('nop')
        return self._tree.get_children_codes(ancestor, level)

    def get_travel_path(self, child, ancestor):
        if False:
            print('Hello World!')
        res = []
        while child > ancestor:
            res.append(child)
            child = int((child - 1) / self._branch)
        return res

    def get_pi_relation(self, ids, level):
        if False:
            return 10
        codes = self.get_ancestor_codes(ids, level)
        return dict(zip(ids, codes))

    def init_layerwise_sampler(self, layer_sample_counts, start_sample_layer=1, seed=0):
        if False:
            while True:
                i = 10
        assert self._layerwise_sampler is None
        self._layerwise_sampler = core.IndexSampler('by_layerwise', self._name)
        self._layerwise_sampler.init_layerwise_conf(layer_sample_counts, start_sample_layer, seed)

    def layerwise_sample(self, user_input, index_input, with_hierarchy=False):
        if False:
            while True:
                i = 10
        if self._layerwise_sampler is None:
            raise ValueError('please init layerwise_sampler first.')
        return self._layerwise_sampler.sample(user_input, index_input, with_hierarchy)