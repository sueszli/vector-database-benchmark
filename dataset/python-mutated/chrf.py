from dataclasses import dataclass
from fairseq.dataclass import FairseqDataclass
from fairseq.scoring import BaseScorer, register_scorer

@dataclass
class ChrFScorerConfig(FairseqDataclass):
    pass

@register_scorer('chrf', dataclass=ChrFScorerConfig)
class ChrFScorer(BaseScorer):

    def __init__(self, args):
        if False:
            for i in range(10):
                print('nop')
        super(ChrFScorer, self).__init__(args)
        import sacrebleu
        self.sacrebleu = sacrebleu

    def add_string(self, ref, pred):
        if False:
            return 10
        self.ref.append(ref)
        self.pred.append(pred)

    def score(self, order=4):
        if False:
            print('Hello World!')
        return self.result_string(order).score

    def result_string(self, order=4):
        if False:
            print('Hello World!')
        if order != 4:
            raise NotImplementedError
        return self.sacrebleu.corpus_chrf(self.pred, [self.ref]).format()