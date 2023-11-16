import pinecone
import os, time, uuid
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader

class customVectorDB:
    """
    The custom VectorDB implementation behind pinecone to support the chatbot.

    Key features:
    (1) A combination of both local context and global context.
    (2) Data retrieval is not based on user inputs; LLM help to generate the actual retrieval with the embedding query.


    Functionalities:
    (1) Store information into the vectorDB
    (2) Retrieve information from the vectorDB

    """

    def __init__(self, project_name: str, vectordb_name: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize the vectorDB with the project name.\n        :param project_name: the unique identifier for the project. It should be the project name.\n        :param file_name: the file name to be stored into the vectorDB. It must be provided for proper initialization.\n        :param vectordb_name: the name of the vectorDB. It should be the name of the vectorDB to use.\n        '
        assert project_name != ''
        self.project_name = project_name
        pinecone_api_key = os.getenv('PINECONE_API_KEY', None)
        os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY', None)
        self.vectordb_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), vectordb_name)
        if not os.path.exists(self.vectordb_directory):
            os.mkdir(self.vectordb_directory)
        self.uuid = str(uuid.uuid4())
        self.local_context_directory = os.path.join(self.vectordb_directory, self.project_name + '_' + self.uuid)
        if not os.path.exists(self.local_context_directory):
            os.mkdir(self.local_context_directory)
        pinecone.init(api_key=pinecone_api_key, environment='gcp-starter')
        if self.project_name not in pinecone.list_indexes():
            pinecone.create_index(name=self.project_name, metric='cosine', dimension=1536)
        self.vectorDB = Pinecone.from_existing_index(self.project_name, OpenAIEmbeddings())

    def __del__(self):
        if False:
            print('Hello World!')
        '\n        TODO: Consider deleting the vectorDB. For now just keep the contents in the index.\n        :return:\n        '
        pass

    def _save_text(self, _text: str) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Handler function that saves everything into the temporary folder.\n        :param _text:\n        :return:\n        '
        filename = str(uuid.uuid4()) + '.txt'
        with open(os.path.join(self.local_context_directory, filename), 'w') as f:
            f.write(_text)
        return os.path.join(self.local_context_directory, filename)

    def store_file(self, filename: str, metadata: [dict]=None):
        if False:
            i = 10
            return i + 15
        '\n        Store the file into the vectorDB. Use `Pinecone.add_texts`\n        :param filename: the filename of the file to be stored.\n        :param metadata: the metadata of the file to be stored. It is a list of\n        :return: None\n        '
        loader = TextLoader(filename)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        self.vectorDB.add_texts([t.page_content for t in texts])

    def store_text(self, content: str, metadata: [dict]=None):
        if False:
            return 10
        '\n        Store the text into the vectorDB. Use `Pinecone.add_texts`\n        :param content: the text to be stored.\n        :return: None\n        '
        filename = self._save_text(content)
        self.store_file(filename, metadata=metadata)

    def retrieval(self, keyword: str, metadata: [dict]=None) -> [dict]:
        if False:
            return 10
        '\n        Retrieve the information from the vectorDB.\n        :param keyword: the keyword to be retrieved.\n        :param metadata: the metadata of the keyword to be retrieved.\n        :return: the retrieval result.\n        '
        retrieval_result = self.vectorDB.similarity_search(keyword)
        return retrieval_result

    def delete_index(self):
        if False:
            print('Hello World!')
        '\n        Delete the index from the pinecone.\n        :return: None\n        '
        pinecone.delete_index(name=self.project_name)