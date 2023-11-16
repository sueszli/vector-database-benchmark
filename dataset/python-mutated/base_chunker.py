import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embedchain.config.add_config import ChunkerConfig
from embedchain.helper.json_serializable import JSONSerializable
from embedchain.models.data_type import DataType

class BaseChunker(JSONSerializable):

    def __init__(self, text_splitter):
        if False:
            i = 10
            return i + 15
        'Initialize the chunker.'
        if text_splitter is None:
            config = ChunkerConfig(chunk_size=1000, chunk_overlap=0, length_function=len)
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap, length_function=config.length_function)
        else:
            self.text_splitter = text_splitter
        self.data_type = None

    def create_chunks(self, loader, src, app_id=None):
        if False:
            i = 10
            return i + 15
        "\n        Loads data and chunks it.\n\n        :param loader: The loader which's `load_data` method is used to create\n        the raw data.\n        :param src: The data to be handled by the loader. Can be a URL for\n        remote sources or local content for local loaders.\n        :param app_id: App id used to generate the doc_id.\n        "
        documents = []
        chunk_ids = []
        idMap = {}
        data_result = loader.load_data(src)
        data_records = data_result['data']
        doc_id = data_result['doc_id']
        doc_id = f'{app_id}--{doc_id}' if app_id is not None else doc_id
        metadatas = []
        for data in data_records:
            content = data['content']
            meta_data = data['meta_data']
            meta_data['data_type'] = self.data_type.value
            meta_data['doc_id'] = doc_id
            url = meta_data['url']
            chunks = self.get_chunks(content)
            for chunk in chunks:
                chunk_id = hashlib.sha256((chunk + url).encode()).hexdigest()
                chunk_id = f'{app_id}--{chunk_id}' if app_id is not None else chunk_id
                if idMap.get(chunk_id) is None:
                    idMap[chunk_id] = True
                    chunk_ids.append(chunk_id)
                    documents.append(chunk)
                    metadatas.append(meta_data)
        return {'documents': documents, 'ids': chunk_ids, 'metadatas': metadatas, 'doc_id': doc_id}

    def get_chunks(self, content):
        if False:
            i = 10
            return i + 15
        '\n        Returns chunks using text splitter instance.\n\n        Override in child class if custom logic.\n        '
        return self.text_splitter.split_text(content)

    def set_data_type(self, data_type: DataType):
        if False:
            return 10
        '\n        set the data type of chunker\n        '
        self.data_type = data_type

    def get_word_count(self, documents):
        if False:
            return 10
        return sum([len(document.split(' ')) for document in documents])