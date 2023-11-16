import logging
from pathlib import Path
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack.nodes.file_classifier import FileTypeClassifier
from haystack.nodes.file_converter import TextConverter
from haystack.nodes.preprocessor import PreProcessor
from haystack.pipelines import Pipeline
from haystack.utils import fetch_archive_from_http, launch_es, print_answers
logging.basicConfig(format='%(levelname)s - %(name)s -  %(message)s', level=logging.WARNING)
logging.getLogger('haystack').setLevel(logging.INFO)

def basic_qa_pipeline():
    if False:
        return 10
    document_store = ElasticsearchDocumentStore(host='localhost', username='', password='', index='example-document')
    doc_dir = 'data/basic_qa_pipeline'
    s3_url = 'https://core-engineering.s3.eu-central-1.amazonaws.com/public/scripts/wiki_gameofthrones_txt1.zip'
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)
    file_paths = [p for p in Path(doc_dir).glob('**/*')]
    files_metadata = [{'name': path.name} for path in file_paths]
    indexing_pipeline = Pipeline()
    classifier = FileTypeClassifier()
    indexing_pipeline.add_node(classifier, name='Classifier', inputs=['File'])
    text_converter = TextConverter(remove_numeric_tables=True)
    indexing_pipeline.add_node(text_converter, name='Text_converter', inputs=['Classifier.output_1'])
    preprocessor = PreProcessor(clean_whitespace=True, clean_empty_lines=True, split_length=100, split_overlap=50, split_respect_sentence_boundary=True)
    indexing_pipeline.add_node(preprocessor, name='Preprocessor', inputs=['Text_converter'])
    indexing_pipeline.add_node(document_store, name='Document_Store', inputs=['Preprocessor'])
    indexing_pipeline.run(file_paths=file_paths, meta=files_metadata)
    retriever = BM25Retriever(document_store=document_store)
    reader = FARMReader(model_name_or_path='deepset/roberta-base-squad2', use_gpu=True)
    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name='Retriever', inputs=['Query'])
    pipeline.add_node(component=reader, name='Reader', inputs=['Retriever'])
    prediction = pipeline.run(query='Who is the father of Arya Stark?', params={'Retriever': {'top_k': 10}, 'Reader': {'top_k': 5}})
    print_answers(prediction, details='minimum')
    document_store.delete_index(index='example-document')
    return prediction
if __name__ == '__main__':
    launch_es()
    basic_qa_pipeline()