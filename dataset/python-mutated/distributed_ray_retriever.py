import logging
import random
import ray
from transformers import RagConfig, RagRetriever, RagTokenizer
from transformers.models.rag.retrieval_rag import CustomHFIndex
logger = logging.getLogger(__name__)

class RayRetriever:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.initialized = False

    def create_rag_retriever(self, config, question_encoder_tokenizer, generator_tokenizer, index):
        if False:
            for i in range(10):
                print('nop')
        if not self.initialized:
            self.retriever = RagRetriever(config, question_encoder_tokenizer=question_encoder_tokenizer, generator_tokenizer=generator_tokenizer, index=index, init_retrieval=False)
            self.initialized = True

    def init_retrieval(self):
        if False:
            return 10
        self.retriever.index.init_index()

    def retrieve(self, question_hidden_states, n_docs):
        if False:
            while True:
                i = 10
        (doc_ids, retrieved_doc_embeds) = self.retriever._main_retrieve(question_hidden_states, n_docs)
        return (doc_ids, retrieved_doc_embeds)

class RagRayDistributedRetriever(RagRetriever):
    """
    A distributed retriever built on top of the ``Ray`` API, a library
    for building distributed applications (https://docs.ray.io/en/master/).
    package. During training, all training workers initialize their own
    instance of a `RagRayDistributedRetriever`, and each instance of
    this distributed retriever shares a common set of Retrieval Ray
    Actors (https://docs.ray.io/en/master/walkthrough.html#remote
    -classes-actors) that load the index on separate processes. Ray
    handles the communication between the `RagRayDistributedRetriever`
    instances and the remote Ray actors. If training is done in a
    non-distributed setup, the index will simply be loaded in the same
    process as the training worker and Ray will not be used.

    Args:
        config (:class:`~transformers.RagConfig`):
            The configuration of the RAG model this Retriever is used with. Contains parameters indicating which ``Index`` to build.
        question_encoder_tokenizer (:class:`~transformers.PreTrainedTokenizer`):
            The tokenizer that was used to tokenize the question.
            It is used to decode the question and then use the generator_tokenizer.
        generator_tokenizer (:class:`~transformers.PreTrainedTokenizer`):
            The tokenizer used for the generator part of the RagModel.
        retrieval_workers (:obj:`List[ray.ActorClass(RayRetriever)]`): A list of already initialized `RayRetriever` actors.
            These actor classes run on remote processes and are responsible for performing the index lookup.
        index (:class:`~transformers.retrieval_rag.Index`, optional, defaults to the one defined by the configuration):
            If specified, use this index instead of the one built using the configuration
    """

    def __init__(self, config, question_encoder_tokenizer, generator_tokenizer, retrieval_workers, index=None):
        if False:
            for i in range(10):
                print('nop')
        if index is not None and index.is_initialized() and (len(retrieval_workers) > 0):
            raise ValueError("When using Ray for distributed fine-tuning, you'll need to provide the paths instead, as the dataset and the index are loaded separately. More info in examples/rag/use_own_knowledge_dataset.py ")
        super().__init__(config, question_encoder_tokenizer=question_encoder_tokenizer, generator_tokenizer=generator_tokenizer, index=index, init_retrieval=False)
        self.retrieval_workers = retrieval_workers
        if len(self.retrieval_workers) > 0:
            ray.get([worker.create_rag_retriever.remote(config, question_encoder_tokenizer, generator_tokenizer, index) for worker in self.retrieval_workers])

    def init_retrieval(self):
        if False:
            while True:
                i = 10
        '\n        Retriever initialization function, needs to be called from the\n        training process. This function triggers retrieval initialization\n        for all retrieval actors if using distributed setting, or loads\n        index into current process if training is not distributed.\n        '
        logger.info('initializing retrieval')
        if len(self.retrieval_workers) > 0:
            ray.get([worker.init_retrieval.remote() for worker in self.retrieval_workers])
        else:
            self.index.init_index()

    def retrieve(self, question_hidden_states, n_docs):
        if False:
            print('Hello World!')
        '\n        Retrieves documents for specified ``question_hidden_states``. If\n        running training with multiple workers, a random retrieval actor is\n        selected to perform the index lookup and return the result.\n\n        Args:\n            question_hidden_states (:obj:`np.ndarray` of shape :obj:`(batch_size, vector_size)`):\n                A batch of query vectors to retrieve with.\n            n_docs (:obj:`int`):\n                The number of docs retrieved per query.\n\n        Output:\n            retrieved_doc_embeds (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs, dim)`\n                The retrieval embeddings of the retrieved docs per query.\n            doc_ids (:obj:`np.ndarray` of shape :obj:`batch_size, n_docs`)\n                The ids of the documents in the index\n            doc_dicts (:obj:`List[dict]`):\n                The retrieved_doc_embeds examples per query.\n        '
        if len(self.retrieval_workers) > 0:
            random_worker = self.retrieval_workers[random.randint(0, len(self.retrieval_workers) - 1)]
            (doc_ids, retrieved_doc_embeds) = ray.get(random_worker.retrieve.remote(question_hidden_states, n_docs))
        else:
            (doc_ids, retrieved_doc_embeds) = self._main_retrieve(question_hidden_states, n_docs)
        return (retrieved_doc_embeds, doc_ids, self.index.get_doc_dicts(doc_ids))

    @classmethod
    def get_tokenizers(cls, retriever_name_or_path, indexed_dataset=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return super(RagRayDistributedRetriever, cls).get_tokenizers(retriever_name_or_path, indexed_dataset, **kwargs)

    @classmethod
    def from_pretrained(cls, retriever_name_or_path, actor_handles, indexed_dataset=None, **kwargs):
        if False:
            print('Hello World!')
        config = kwargs.pop('config', None) or RagConfig.from_pretrained(retriever_name_or_path, **kwargs)
        rag_tokenizer = RagTokenizer.from_pretrained(retriever_name_or_path, config=config)
        question_encoder_tokenizer = rag_tokenizer.question_encoder
        generator_tokenizer = rag_tokenizer.generator
        if indexed_dataset is not None:
            config.index_name = 'custom'
            index = CustomHFIndex(config.retrieval_vector_size, indexed_dataset)
        else:
            index = cls._build_index(config)
        return cls(config, question_encoder_tokenizer=question_encoder_tokenizer, generator_tokenizer=generator_tokenizer, retrieval_workers=actor_handles, index=index)