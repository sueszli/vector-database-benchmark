import logging
from collections import defaultdict
from math import inf
from typing import List, Optional
from haystack.nodes.other.join import JoinNode
from haystack.schema import Document
logger = logging.getLogger(__name__)

class JoinDocuments(JoinNode):
    """
    A node to join documents outputted by multiple retriever nodes.

    The node allows multiple join modes:
    * concatenate: combine the documents from multiple nodes.
                   In case of duplicate documents, the one with the highest score is kept.
    * merge: merge scores of documents from multiple nodes. Optionally, each input score can be given a different
             `weight` & a `top_k` limit can be set. This mode can also be used for "reranking" retrieved documents.
    * reciprocal_rank_fusion: combines the documents based on their rank in multiple nodes.
    """
    outgoing_edges = 1

    def __init__(self, join_mode: str='concatenate', weights: Optional[List[float]]=None, top_k_join: Optional[int]=None, sort_by_score: bool=True):
        if False:
            i = 10
            return i + 15
        '\n        :param join_mode: `concatenate` to combine documents from multiple retrievers `merge` to aggregate scores of\n                          individual documents, `reciprocal_rank_fusion` to apply rank based scoring.\n        :param weights: A node-wise list(length of list must be equal to the number of input nodes) of weights for\n                        adjusting document scores when using the `merge` join_mode. By default, equal weight is given\n                        to each retriever score. This param is not compatible with the `concatenate` join_mode.\n        :param top_k_join: Limit documents to top_k based on the resulting scores of the join.\n        :param sort_by_score: Whether to sort the incoming documents by their score. Set this to True if all your\n                              Documents are coming with `score` values. Set to False if any of the Documents come\n                              from sources where the `score` is set to `None`, like `TfidfRetriever` on Elasticsearch.\n        '
        assert join_mode in ['concatenate', 'merge', 'reciprocal_rank_fusion'], f"JoinDocuments node does not support '{join_mode}' join_mode."
        assert not (weights is not None and join_mode == 'concatenate'), "Weights are not compatible with 'concatenate' join_mode."
        super().__init__()
        self.join_mode = join_mode
        self.weights = [float(i) / sum(weights) for i in weights] if weights else None
        self.top_k_join = top_k_join
        self.sort_by_score = sort_by_score

    def run_accumulated(self, inputs: List[dict], top_k_join: Optional[int]=None):
        if False:
            while True:
                i = 10
        results = [inp['documents'] for inp in inputs]
        document_map = {doc.id: doc for result in results for doc in result}
        if self.join_mode == 'concatenate':
            scores_map = self._concatenate_results(results, document_map)
        elif self.join_mode == 'merge':
            scores_map = self._calculate_comb_sum(results)
        elif self.join_mode == 'reciprocal_rank_fusion':
            scores_map = self._calculate_rrf(results)
        else:
            raise ValueError(f'Invalid join_mode: {self.join_mode}')
        if self.sort_by_score:
            sorted_docs = sorted(scores_map.items(), key=lambda d: d[1] if d[1] is not None else -inf, reverse=True)
            if any((s is None for s in scores_map.values())):
                logger.info('The `JoinDocuments` node has received some documents with `score=None` - and was requested to sort the documents by score, so the `score=None` documents got sorted as if their score would be `-infinity`.')
        else:
            sorted_docs = list(scores_map.items())
        if not top_k_join:
            top_k_join = self.top_k_join
        if not top_k_join:
            top_k_join = len(sorted_docs)
        docs = []
        for (id, score) in sorted_docs[:top_k_join]:
            doc = document_map[id]
            doc.score = score
            docs.append(doc)
        output = {'documents': docs, 'labels': inputs[0].get('labels', None)}
        return (output, 'output_1')

    def run_batch_accumulated(self, inputs: List[dict], top_k_join: Optional[int]=None):
        if False:
            i = 10
            return i + 15
        if isinstance(inputs[0]['documents'][0], Document):
            return self.run(inputs=inputs, top_k_join=top_k_join)
        else:
            output_docs = []
            incoming_edges = [inp['documents'] for inp in inputs]
            for idx in range(len(incoming_edges[0])):
                cur_docs_to_join = []
                for edge in incoming_edges:
                    cur_docs_to_join.append({'documents': edge[idx]})
                (cur, _) = self.run(inputs=cur_docs_to_join, top_k_join=top_k_join)
                output_docs.append(cur['documents'])
            output = {'documents': output_docs, 'labels': inputs[0].get('labels', None)}
            return (output, 'output_1')

    def _concatenate_results(self, results, document_map):
        if False:
            print('Hello World!')
        '\n        Concatenates multiple document result lists.\n        Return the documents with the higher score.\n        '
        list_id = list(document_map.keys())
        scores_map = {}
        for idx in list_id:
            tmp = []
            for result in results:
                for doc in result:
                    if doc.id == idx:
                        tmp.append(doc)
            item_best_score = max(tmp, key=lambda x: x.score if x.score is not None else -inf)
            scores_map.update({idx: item_best_score.score})
        return scores_map

    def _calculate_comb_sum(self, results):
        if False:
            while True:
                i = 10
        '\n        Calculates a combination sum by multiplying each score by its weight.\n        '
        scores_map = defaultdict(int)
        weights = self.weights if self.weights else [1 / len(results)] * len(results)
        for (result, weight) in zip(results, weights):
            for doc in result:
                scores_map[doc.id] += (doc.score if doc.score else 0) * weight
        return scores_map

    def _calculate_rrf(self, results):
        if False:
            while True:
                i = 10
        '\n        Calculates the reciprocal rank fusion. The constant K is set to 61 (60 was suggested by the original paper,\n        plus 1 as python lists are 0-based and the paper used 1-based ranking).\n        '
        K = 61
        scores_map = defaultdict(int)
        for result in results:
            for (rank, doc) in enumerate(result):
                scores_map[doc.id] += 1 / (K + rank)
        return scores_map