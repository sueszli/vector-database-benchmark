import torch
from typing import List, Dict
from .base_decoder import BaseDecoder

class ViterbiDecoder(BaseDecoder):

    def decode(self, emissions: torch.FloatTensor) -> List[List[Dict[str, torch.LongTensor]]]:
        if False:
            print('Hello World!')

        def get_pred(e):
            if False:
                for i in range(10):
                    print('nop')
            toks = e.argmax(dim=-1).unique_consecutive()
            return toks[toks != self.blank]
        return [[{'tokens': get_pred(x), 'score': 0}] for x in emissions]