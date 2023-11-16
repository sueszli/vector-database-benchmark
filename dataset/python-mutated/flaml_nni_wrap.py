from flaml.tune.searcher.blendsearch import BlendSearchTuner as BST

class BlendSearchTuner(BST):

    def __init__(self, low_cost_partial_config={'hidden_size': 128}):
        if False:
            i = 10
            return i + 15
        super.__init__(self, low_cost_partial_config=low_cost_partial_config)