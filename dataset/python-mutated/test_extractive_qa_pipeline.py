import json
from haystack.preview import Pipeline, Document
from haystack.preview.document_stores import InMemoryDocumentStore
from haystack.preview.components.retrievers import InMemoryBM25Retriever
from haystack.preview.components.readers import ExtractiveReader

def test_extractive_qa_pipeline(tmp_path):
    if False:
        i = 10
        return i + 15
    qa_pipeline = Pipeline()
    qa_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=InMemoryDocumentStore()), name='retriever')
    qa_pipeline.add_component(instance=ExtractiveReader(model_name_or_path='deepset/tinyroberta-squad2'), name='reader')
    qa_pipeline.connect('retriever', 'reader')
    qa_pipeline.draw(tmp_path / 'test_extractive_qa_pipeline.png')
    with open(tmp_path / 'test_bm25_rag_pipeline.json', 'w') as f:
        print(json.dumps(qa_pipeline.to_dict(), indent=4))
        json.dump(qa_pipeline.to_dict(), f)
    with open(tmp_path / 'test_bm25_rag_pipeline.json', 'r') as f:
        qa_pipeline = Pipeline.from_dict(json.load(f))
    documents = [Document(content='My name is Jean and I live in Paris.'), Document(content='My name is Mark and I live in Berlin.'), Document(content='My name is Giorgio and I live in Rome.')]
    qa_pipeline.get_component('retriever').document_store.write_documents(documents)
    questions = ['Who lives in Paris?', 'Who lives in Berlin?', 'Who lives in Rome?']
    answers_spywords = ['Jean', 'Mark', 'Giorgio']
    for (question, spyword, doc) in zip(questions, answers_spywords, documents):
        result = qa_pipeline.run({'retriever': {'query': question}, 'reader': {'query': question}})
        extracted_answers = result['reader']['answers']
        assert len(extracted_answers) > 1
        assert spyword in extracted_answers[0].data
        assert extracted_answers[-1].data is None
        assert extracted_answers[0].probability >= extracted_answers[-1].probability
        for answer in extracted_answers:
            assert answer.query == question
            assert hasattr(answer, 'probability')
            assert hasattr(answer, 'start')
            assert hasattr(answer, 'end')
            assert hasattr(answer, 'document')
            if answer.document is not None:
                assert answer.document == doc