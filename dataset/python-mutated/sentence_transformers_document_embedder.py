from typing import List, Optional, Union, Dict, Any
from haystack.preview import component, Document, default_to_dict
from haystack.preview.components.embedders.backends.sentence_transformers_backend import _SentenceTransformersEmbeddingBackendFactory

@component
class SentenceTransformersDocumentEmbedder:
    """
    A component for computing Document embeddings using Sentence Transformers models.
    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    from haystack.preview import Document
    from haystack.preview.components.embedders import SentenceTransformersDocumentEmbedder
    doc = Document(text="I love pizza!")
    doc_embedder = SentenceTransformersDocumentEmbedder()
    doc_embedder.warm_up()

    result = doc_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [-0.07804739475250244, 0.1498992145061493, ...]
    ```
    """

    def __init__(self, model_name_or_path: str='sentence-transformers/all-mpnet-base-v2', device: Optional[str]=None, token: Union[bool, str, None]=None, prefix: str='', suffix: str='', batch_size: int=32, progress_bar: bool=True, normalize_embeddings: bool=False, metadata_fields_to_embed: Optional[List[str]]=None, embedding_separator: str='\n'):
        if False:
            i = 10
            return i + 15
        "\n        Create a SentenceTransformersDocumentEmbedder component.\n\n        :param model_name_or_path: Local path or name of the model in Hugging Face's model hub,\n            such as ``'sentence-transformers/all-mpnet-base-v2'``.\n        :param device: Device (like 'cuda' / 'cpu') that should be used for computation.\n            Defaults to CPU.\n        :param token: The API token used to download private models from Hugging Face.\n            If this parameter is set to `True`, then the token generated when running\n            `transformers-cli login` (stored in ~/.huggingface) will be used.\n        :param prefix: A string to add to the beginning of each Document text before embedding.\n            Can be used to prepend the text with an instruction, as required by some embedding models,\n            such as E5 and bge.\n        :param suffix: A string to add to the end of each Document text before embedding.\n        :param batch_size: Number of strings to encode at once.\n        :param progress_bar: If true, displays progress bar during embedding.\n        :param normalize_embeddings: If set to true, returned vectors will have length 1.\n        :param metadata_fields_to_embed: List of meta fields that should be embedded along with the Document content.\n        :param embedding_separator: Separator used to concatenate the meta fields to the Document content.\n        "
        self.model_name_or_path = model_name_or_path
        self.device = device or 'cpu'
        self.token = token
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.normalize_embeddings = normalize_embeddings
        self.metadata_fields_to_embed = metadata_fields_to_embed or []
        self.embedding_separator = embedding_separator

    def _get_telemetry_data(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Data that is sent to Posthog for usage analytics.\n        '
        return {'model': self.model_name_or_path}

    def to_dict(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        '\n        Serialize this component to a dictionary.\n        '
        return default_to_dict(self, model_name_or_path=self.model_name_or_path, device=self.device, token=self.token if not isinstance(self.token, str) else None, prefix=self.prefix, suffix=self.suffix, batch_size=self.batch_size, progress_bar=self.progress_bar, normalize_embeddings=self.normalize_embeddings, metadata_fields_to_embed=self.metadata_fields_to_embed, embedding_separator=self.embedding_separator)

    def warm_up(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Load the embedding backend.\n        '
        if not hasattr(self, 'embedding_backend'):
            self.embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(model_name_or_path=self.model_name_or_path, device=self.device, use_auth_token=self.token)

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        if False:
            i = 10
            return i + 15
        '\n        Embed a list of Documents.\n        The embedding of each Document is stored in the `embedding` field of the Document.\n        '
        if not isinstance(documents, list) or (documents and (not isinstance(documents[0], Document))):
            raise TypeError('SentenceTransformersDocumentEmbedder expects a list of Documents as input.In case you want to embed a list of strings, please use the SentenceTransformersTextEmbedder.')
        if not hasattr(self, 'embedding_backend'):
            raise RuntimeError('The embedding model has not been loaded. Please call warm_up() before running.')
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [str(doc.meta[key]) for key in self.metadata_fields_to_embed if key in doc.meta and doc.meta[key]]
            text_to_embed = self.prefix + self.embedding_separator.join(meta_values_to_embed + [doc.content or '']) + self.suffix
            texts_to_embed.append(text_to_embed)
        embeddings = self.embedding_backend.embed(texts_to_embed, batch_size=self.batch_size, show_progress_bar=self.progress_bar, normalize_embeddings=self.normalize_embeddings)
        for (doc, emb) in zip(documents, embeddings):
            doc.embedding = emb
        return {'documents': documents}