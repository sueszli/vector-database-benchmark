class LabelSchema:

    def __init__(self, labels):
        if False:
            print('Hello World!')
        self.labels = {self.preprocess(val): idx for (idx, val) in enumerate(labels)}
        self.idx = {idx: self.preprocess(val) for (idx, val) in enumerate(labels)}

    def get_id(self, label):
        if False:
            while True:
                i = 10
        if self.preprocess(label) in self.labels:
            return self.labels[self.preprocess(label)]
        return None

    def preprocess(self, item):
        if False:
            return 10
        return item.lower()

class SNLILabelSchema(LabelSchema):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(SNLILabelSchema, self).__init__(['neither', 'contradiction', 'entailment'])