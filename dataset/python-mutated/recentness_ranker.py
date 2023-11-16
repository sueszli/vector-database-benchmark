import logging
import warnings
from collections import defaultdict
from typing import List, Union, Optional, Dict, Literal
from dateutil.parser import parse, ParserError
from haystack.errors import NodeError
from haystack.nodes.ranker.base import BaseRanker
from haystack.schema import Document
logger = logging.getLogger(__name__)

class RecentnessRanker(BaseRanker):
    outgoing_edges = 1

    def __init__(self, date_meta_field: str, weight: float=0.5, top_k: Optional[int]=None, ranking_mode: Literal['reciprocal_rank_fusion', 'score']='reciprocal_rank_fusion'):
        if False:
            return 10
        "\n        This Node is used to rerank retrieved documents based on their age. Newer documents will rank higher.\n        The importance of recentness is parametrized through the weight parameter.\n\n        :param date_meta_field: Identifier pointing to the date field in the metadata.\n                This is a required parameter, since we need dates for sorting.\n        :param weight: in range [0,1].\n                0 disables sorting by age.\n                0.5 content and age have the same impact.\n                1 means sorting only by age, most recent comes first.\n        :param top_k: (optional) How many documents to return. If not provided, all documents will be returned.\n                It can make sense to have large top-k values from the initial retrievers and filter docs down in the\n                RecentnessRanker with this top_k parameter.\n        :param ranking_mode: The mode used to combine retriever and recentness. Possible values are 'reciprocal_rank_fusion' (default) and 'score'.\n                Make sure to use 'score' mode only with retrievers/rankers that give back OK score in range [0,1].\n        "
        super().__init__()
        self.date_meta_field = date_meta_field
        self.weight = weight
        self.top_k = top_k
        self.ranking_mode = ranking_mode
        if self.weight < 0 or self.weight > 1:
            raise NodeError("\n                Param <weight> needs to be in range [0,1] but was set to '{}'.\n\n                '0' disables sorting by recency, '0.5' gives equal weight to previous relevance scores and recency, and '1' ranks by recency only.\n\n                Please change param <weight> when initializing the RecentnessRanker.\n                ".format(self.weight))

    def predict(self, query: str, documents: List[Document], top_k: Optional[int]=None) -> List[Document]:
        if False:
            print('Hello World!')
        '\n        This method is used to rank a list of documents based on their age and relevance by:\n        1. Adjusting the relevance score from the previous node (or, for RRF, calculating it from scratch, then adjusting) based on the chosen weight in initial parameters.\n        2. Sorting the documents based on their age in the metadata, calculating the recentness score, adjusting it by weight as well.\n        3. Returning top-k documents (or all, if top-k not provided) in the documents dictionary sorted by final score (relevance score + recentness score).\n\n        :param query: Not used in practice (so can be left blank), as this ranker does not perform sorting based on semantic closeness of documents to the query.\n        :param documents: Documents provided for ranking.\n        :param top_k: (optional) How many documents to return at the end. If not provided, all documents will be returned, sorted by relevance and recentness (adjusted by weight).\n        '
        try:
            sorted_by_date = sorted(documents, reverse=True, key=lambda x: parse(x.meta[self.date_meta_field]))
        except KeyError:
            raise NodeError("\n                Param <date_meta_field> was set to '{}', but document(s) {} do not contain this metadata key.\n\n                Please double-check the names of existing metadata fields of your documents \n\n                and set <date_meta_field> to the name of the field that contains dates.\n                ".format(self.date_meta_field, ','.join([doc.id for doc in documents if self.date_meta_field not in doc.meta])))
        except ParserError:
            logger.error('\n                Could not parse date information for dates: %s\n\n                Continuing without sorting by date.\n                ', ' - '.join([doc.meta.get(self.date_meta_field, 'identifier wrong') for doc in documents]))
            return documents
        scores_map: Dict = defaultdict(int)
        if self.ranking_mode not in ['reciprocal_rank_fusion', 'score']:
            raise NodeError("\n                Param <ranking_mode> needs to be 'reciprocal_rank_fusion' or 'score' but was set to '{}'. \n\n                Please change the <ranking_mode> when initializing the RecentnessRanker.\n                ".format(self.ranking_mode))
        for (i, doc) in enumerate(documents):
            if self.ranking_mode == 'reciprocal_rank_fusion':
                scores_map[doc.id] += self._calculate_rrf(rank=i) * (1 - self.weight)
            elif self.ranking_mode == 'score':
                score = float(0)
                if doc.score is None:
                    warnings.warn('The score was not provided; defaulting to 0')
                elif doc.score < 0 or doc.score > 1:
                    warnings.warn('The score {} for document {} is outside the [0,1] range; defaulting to 0'.format(doc.score, doc.id))
                else:
                    score = doc.score
                scores_map[doc.id] += score * (1 - self.weight)
        for (i, doc) in enumerate(sorted_by_date):
            if self.ranking_mode == 'reciprocal_rank_fusion':
                scores_map[doc.id] += self._calculate_rrf(rank=i) * self.weight
            elif self.ranking_mode == 'score':
                scores_map[doc.id] += self._calc_recentness_score(rank=i, amount=len(sorted_by_date)) * self.weight
        top_k = top_k or self.top_k or len(documents)
        for doc in documents:
            doc.score = scores_map[doc.id]
        return sorted(documents, key=lambda doc: doc.score if doc.score is not None else -1, reverse=True)[:top_k]

    def predict_batch(self, queries: List[str], documents: Union[List[Document], List[List[Document]]], top_k: Optional[int]=None, batch_size: Optional[int]=None) -> Union[List[Document], List[List[Document]]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        This method is used to rank A) a list or B) a list of lists (in case the previous node is JoinDocuments) of documents based on their age and relevance.\n        In case A, the predict method defined earlier is applied to the provided list.\n        In case B, predict method is applied to each individual list in the list of lists provided, then the results are returned as list of lists.\n\n        :param queries: Not used in practice (so can be left blank), as this ranker does not perform sorting based on semantic closeness of documents to the query.\n        :param documents: Documents provided for ranking in a list or a list of lists.\n        :param top_k: (optional) How many documents to return at the end (per list). If not provided, all documents will be returned, sorted by relevance and recentness (adjusted by weight).\n        :param batch_size:  Not used in practice, so can be left blank.\n        '
        if isinstance(documents[0], Document):
            return self.predict('', documents=documents, top_k=top_k)
        nested_docs = []
        for docs in documents:
            results = self.predict('', documents=docs, top_k=top_k)
            nested_docs.append(results)
        return nested_docs

    @staticmethod
    def _calculate_rrf(rank: int, k: int=61) -> float:
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculates the reciprocal rank fusion. The constant K is set to 61 (60 was suggested by the original paper,\n        plus 1 as python lists are 0-based and the paper [https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf] used 1-based ranking).\n        '
        return 1 / (k + rank)

    @staticmethod
    def _calc_recentness_score(rank: int, amount: int) -> float:
        if False:
            return 10
        '\n        Calculate recentness score as a linear score between most recent and oldest document.\n        This linear scaling is useful to\n          a) reduce the effect of outliers and\n          b) create recentness scoress that are meaningfully distributed in [0,1],\n             similar to scores coming from a retriever/ranker.\n        '
        return (amount - rank) / amount