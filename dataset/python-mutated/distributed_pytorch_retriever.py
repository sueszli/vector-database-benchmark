import logging
import os
from typing import List, Tuple
import numpy as np
import psutil
import torch
import torch.distributed as dist
from transformers import RagRetriever
logger = logging.getLogger(__name__)

class RagPyTorchDistributedRetriever(RagRetriever):
    """
    A distributed retriever built on top of the ``torch.distributed`` communication package. During training all workers
    initialize their own instance of the retriever, however, only the main worker loads the index into memory. The index is stored
    in cpu memory. The index will also work well in a non-distributed setup.

    Args:
        config (:class:`~transformers.RagConfig`):
            The configuration of the RAG model this Retriever is used with. Contains parameters indicating which ``Index`` to build.
        question_encoder_tokenizer (:class:`~transformers.PreTrainedTokenizer`):
            The tokenizer that was used to tokenize the question.
            It is used to decode the question and then use the generator_tokenizer.
        generator_tokenizer (:class:`~transformers.PreTrainedTokenizer`):
            The tokenizer used for the generator part of the RagModel.
        index (:class:`~transformers.models.rag.retrieval_rag.Index`, optional, defaults to the one defined by the configuration):
            If specified, use this index instead of the one built using the configuration
    """

    def __init__(self, config, question_encoder_tokenizer, generator_tokenizer, index=None):
        if False:
            print('Hello World!')
        super().__init__(config, question_encoder_tokenizer=question_encoder_tokenizer, generator_tokenizer=generator_tokenizer, index=index, init_retrieval=False)
        self.process_group = None

    def init_retrieval(self, distributed_port: int):
        if False:
            while True:
                i = 10
        '\n        Retriever initialization function, needs to be called from the training process. The function sets some common parameters\n        and environment variables. On top of that, (only) the main process in the process group loads the index into memory.\n\n        Args:\n            distributed_port (:obj:`int`):\n                The port on which the main communication of the training run is carried out. We set the port for retrieval-related\n                communication as ``distributed_port + 1``.\n        '
        logger.info('initializing retrieval')
        if dist.is_initialized():
            logger.info('dist initialized')
            os.environ['GLOO_SOCKET_IFNAME'] = self._infer_socket_ifname()
            os.environ['MASTER_PORT'] = str(distributed_port + 1)
            self.process_group = dist.new_group(ranks=None, backend='gloo')
        if not dist.is_initialized() or self._is_main():
            logger.info('dist not initialized / main')
            self.index.init_index()
        if dist.is_initialized():
            torch.distributed.barrier(group=self.process_group)

    def _is_main(self):
        if False:
            for i in range(10):
                print('nop')
        return dist.get_rank(group=self.process_group) == 0

    def _scattered(self, scatter_list, target_shape, target_type=torch.float32):
        if False:
            return 10
        target_tensor = torch.empty(target_shape, dtype=target_type)
        dist.scatter(target_tensor, src=0, scatter_list=scatter_list, group=self.process_group)
        return target_tensor

    def _infer_socket_ifname(self):
        if False:
            i = 10
            return i + 15
        addrs = psutil.net_if_addrs()
        ifname = next((addr for addr in addrs if addr.startswith('e')), None)
        return ifname

    def retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, List[dict]]:
        if False:
            return 10
        '\n        Retrieves documents for specified ``question_hidden_states``. The main process, which has the access to the index stored in memory, gathers queries\n        from all the processes in the main training process group, performs the retrieval and scatters back the results.\n\n        Args:\n            question_hidden_states (:obj:`np.ndarray` of shape :obj:`(batch_size, vector_size)`):\n                A batch of query vectors to retrieve with.\n            n_docs (:obj:`int`):\n                The number of docs retrieved per query.\n\n        Output:\n            retrieved_doc_embeds (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs, dim)`\n                The retrieval embeddings of the retrieved docs per query.\n            doc_ids (:obj:`np.ndarray` of shape :obj:`batch_size, n_docs`)\n                The ids of the documents in the index\n            doc_dicts (:obj:`List[dict]`):\n                The retrieved_doc_embeds examples per query.\n        '
        if not dist.is_initialized():
            (doc_ids, retrieved_doc_embeds) = self._main_retrieve(question_hidden_states, n_docs)
            return (retrieved_doc_embeds, doc_ids, self.index.get_doc_dicts(doc_ids))
        world_size = dist.get_world_size(group=self.process_group)
        gather_list = None
        if self._is_main():
            gather_list = [torch.empty(question_hidden_states.shape, dtype=torch.float32) for _ in range(world_size)]
        dist.gather(torch.tensor(question_hidden_states), dst=0, gather_list=gather_list, group=self.process_group)
        n_queries = question_hidden_states.shape[0]
        scatter_ids = []
        scatter_vectors = []
        if self._is_main():
            assert len(gather_list) == world_size
            (ids, vectors) = self._main_retrieve(torch.cat(gather_list).numpy(), n_docs)
            (ids, vectors) = (torch.tensor(ids), torch.tensor(vectors))
            scatter_ids = self._chunk_tensor(ids, n_queries)
            scatter_vectors = self._chunk_tensor(vectors, n_queries)
        doc_ids = self._scattered(scatter_ids, [n_queries, n_docs], target_type=torch.int64)
        retrieved_doc_embeds = self._scattered(scatter_vectors, [n_queries, n_docs, question_hidden_states.shape[1]])
        return (retrieved_doc_embeds.numpy(), doc_ids.numpy(), self.index.get_doc_dicts(doc_ids))