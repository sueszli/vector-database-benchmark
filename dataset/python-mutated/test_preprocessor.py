import sys
from pathlib import Path
from typing import Any, Optional, List
from unittest.mock import Mock
import nltk.data
import pytest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.tmpdir import TempPathFactory
from haystack import Document
from haystack.nodes.file_converter.pdf import PDFToTextConverter
from haystack.nodes.preprocessor.preprocessor import PreProcessor
TEXT = '\nThis is a sample sentence in paragraph_1. This is a sample sentence in paragraph_1. This is a sample sentence in\nparagraph_1. This is a sample sentence in paragraph_1. This is a sample sentence in paragraph_1.\x0c\n\nThis is a sample sentence in paragraph_2. This is a sample sentence in paragraph_2. This is a sample sentence in\nparagraph_2. This is a sample sentence in paragraph_2. This is a sample sentence in paragraph_2.\n\nThis is a sample sentence in paragraph_3. This is a sample sentence in paragraph_3. This is a sample sentence in\nparagraph_3. This is a sample sentence in paragraph_3. This is to trick the test with using an abbreviation\x0c like Dr.\nin the sentence.\n'
HEADLINES = [{'headline': 'sample sentence in paragraph_1', 'start_idx': 11, 'level': 0}, {'headline': 'paragraph_1', 'start_idx': 198, 'level': 1}, {'headline': 'sample sentence in paragraph_2', 'start_idx': 223, 'level': 0}, {'headline': 'in paragraph_2', 'start_idx': 365, 'level': 1}, {'headline': 'sample sentence in paragraph_3', 'start_idx': 434, 'level': 0}, {'headline': 'trick the test', 'start_idx': 603, 'level': 1}]
LEGAL_TEXT_PT = '\nA Lei nº 9.514/1997, que instituiu a alienação fiduciária de\nbens imóveis, é norma especial e posterior ao Código de Defesa do\nConsumidor – CDC. Em tais circunstâncias, o inadimplemento do\ndevedor fiduciante enseja a aplicação da regra prevista nos arts. 26 e 27\nda lei especial” (REsp 1.871.911/SP, rel. Min. Nancy Andrighi, DJe\n25/8/2020).\n\nA Emenda Constitucional n. 35 alterou substancialmente esse mecanismo,\nao determinar, na nova redação conferida ao art. 53: “§ 3º Recebida a\ndenúncia contra o Senador ou Deputado, por crime ocorrido após a\ndiplomação, o Supremo Tribunal Federal dará ciência à Casa respectiva, que,\npor iniciativa de partido político nela representado e pelo voto da maioria de\nseus membros, poderá, até a decisão final, sustar o andamento da ação”.\nVale ressaltar, contudo, que existem, antes do encaminhamento ao\nPresidente da República, os chamados autógrafos. Os autógrafos ocorrem já\ncom o texto definitivamente aprovado pelo Plenário ou pelas comissões,\nquando for o caso. Os autógrafos devem reproduzir com absoluta fidelidade a\nredação final aprovada. O projeto aprovado será encaminhado em autógrafos\nao Presidente da República. O tema encontra-se regulamentado pelo art. 200\ndo RICD e arts. 328 a 331 do RISF.\n'

@pytest.fixture(scope='module')
def module_tmp_dir(tmp_path_factory: TempPathFactory) -> Path:
    if False:
        return 10
    'Module fixture to avoid that the model data is downloaded for each test.'
    return tmp_path_factory.mktemp('nltk_data')

@pytest.fixture(autouse=True)
def patched_nltk_data_path(module_tmp_dir: Path, monkeypatch: MonkeyPatch, tmp_path: Path) -> Path:
    if False:
        for i in range(10):
            print('nop')
    'Patch the NLTK data path to use a temporary directory instead of a local, persistent directory.'
    old_find = nltk.data.find

    def patched_find(resource_name: str, paths: Optional[List[str]]=None) -> str:
        if False:
            for i in range(10):
                print('nop')
        return old_find(resource_name, paths=[str(tmp_path)])
    monkeypatch.setattr(nltk.data, nltk.data.find.__name__, patched_find)
    old_download = nltk.download

    def patched_download(*args: Any, **kwargs: Any) -> bool:
        if False:
            while True:
                i = 10
        return old_download(*args, **kwargs, download_dir=str(tmp_path))
    monkeypatch.setattr(nltk, nltk.download.__name__, patched_download)
    return tmp_path

@pytest.mark.unit
@pytest.mark.parametrize('split_length_and_results', [(1, 15), (10, 2)])
def test_preprocess_sentence_split(split_length_and_results):
    if False:
        for i in range(10):
            print('nop')
    (split_length, expected_documents_count) = split_length_and_results
    document = Document(content=TEXT)
    preprocessor = PreProcessor(split_length=split_length, split_overlap=0, split_by='sentence', split_respect_sentence_boundary=False)
    documents = preprocessor.process(document)
    assert len(documents) == expected_documents_count

@pytest.mark.unit
@pytest.mark.parametrize('split_length_and_results', [(1, 15), (10, 2)])
def test_preprocess_sentence_split_custom_models_wrong_file_format(split_length_and_results, samples_path):
    if False:
        while True:
            i = 10
    (split_length, expected_documents_count) = split_length_and_results
    document = Document(content=TEXT)
    preprocessor = PreProcessor(split_length=split_length, split_overlap=0, split_by='sentence', split_respect_sentence_boundary=False, tokenizer_model_folder=samples_path / 'preprocessor' / 'nltk_models' / 'wrong', language='en')
    documents = preprocessor.process(document)
    assert len(documents) == expected_documents_count

@pytest.mark.unit
@pytest.mark.parametrize('split_length_and_results', [(1, 15), (10, 2)])
def test_preprocess_sentence_split_custom_models_non_default_language(split_length_and_results):
    if False:
        while True:
            i = 10
    (split_length, expected_documents_count) = split_length_and_results
    document = Document(content=TEXT)
    preprocessor = PreProcessor(split_length=split_length, split_overlap=0, split_by='sentence', split_respect_sentence_boundary=False, language='ca')
    documents = preprocessor.process(document)
    assert len(documents) == expected_documents_count

@pytest.mark.unit
@pytest.mark.parametrize('split_length_and_results', [(1, 8), (8, 1)])
def test_preprocess_sentence_split_custom_models(split_length_and_results, samples_path):
    if False:
        print('Hello World!')
    (split_length, expected_documents_count) = split_length_and_results
    document = Document(content=LEGAL_TEXT_PT)
    preprocessor = PreProcessor(split_length=split_length, split_overlap=0, split_by='sentence', split_respect_sentence_boundary=False, language='pt', tokenizer_model_folder=samples_path / 'preprocessor' / 'nltk_models')
    documents = preprocessor.process(document)
    assert len(documents) == expected_documents_count

@pytest.mark.unit
def test_preprocess_word_split():
    if False:
        while True:
            i = 10
    document = Document(content=TEXT)
    preprocessor = PreProcessor(split_length=10, split_overlap=0, split_by='word', split_respect_sentence_boundary=False)
    documents = preprocessor.process(document)
    assert len(documents) == 11
    preprocessor = PreProcessor(split_length=15, split_overlap=0, split_by='word', split_respect_sentence_boundary=True)
    documents = preprocessor.process(document)
    for (i, doc) in enumerate(documents):
        if i == 0:
            assert len(doc.content.split()) == 14
        assert len(doc.content.split()) <= 15 or doc.content.startswith('This is to trick')
    assert len(documents) == 8
    preprocessor = PreProcessor(split_length=40, split_overlap=10, split_by='word', split_respect_sentence_boundary=True)
    documents = preprocessor.process(document)
    assert len(documents) == 5
    preprocessor = PreProcessor(split_length=5, split_overlap=0, split_by='word', split_respect_sentence_boundary=True)
    documents = preprocessor.process(document)
    assert len(documents) == 15

@pytest.mark.unit
@pytest.mark.parametrize('split_length_and_results', [(1, 3), (2, 2)])
def test_preprocess_passage_split(split_length_and_results):
    if False:
        return 10
    (split_length, expected_documents_count) = split_length_and_results
    document = Document(content=TEXT)
    preprocessor = PreProcessor(split_length=split_length, split_overlap=0, split_by='passage', split_respect_sentence_boundary=False)
    documents = preprocessor.process(document)
    assert len(documents) == expected_documents_count

@pytest.mark.skipif(sys.platform in ['win32', 'cygwin'], reason='FIXME Footer not detected correctly on Windows')
def test_clean_header_footer(samples_path):
    if False:
        return 10
    converter = PDFToTextConverter()
    document = converter.convert(file_path=Path(samples_path / 'pdf' / 'sample_pdf_2.pdf'))
    preprocessor = PreProcessor(clean_header_footer=True, split_by=None)
    documents = preprocessor.process(document)
    assert len(documents) == 1
    assert 'This is a header.' not in documents[0].content
    assert 'footer' not in documents[0].content

@pytest.mark.unit
def test_remove_substrings():
    if False:
        for i in range(10):
            print('nop')
    document = Document(content='This is a header. Some additional text. wiki. Some emoji ✨ 🪲 Weird whitespace\x08\x08\x08.')
    assert 'This is a header.' in document.content
    assert 'wiki' in document.content
    assert '🪲' in document.content
    assert 'whitespace' in document.content
    assert '✨' in document.content
    preprocessor = PreProcessor(remove_substrings=['This is a header.', 'wiki', '🪲'])
    documents = preprocessor.process(document)
    assert 'This is a header.' not in documents[0].content
    assert 'wiki' not in documents[0].content
    assert '🪲' not in documents[0].content
    assert 'whitespace' in documents[0].content
    assert '✨' in documents[0].content

@pytest.mark.unit
def test_id_hash_keys_from_pipeline_params():
    if False:
        print('Hello World!')
    document_1 = Document(content='This is a document.', meta={'key': 'a'})
    document_2 = Document(content='This is a document.', meta={'key': 'b'})
    assert document_1.id == document_2.id
    preprocessor = PreProcessor(split_length=2, split_respect_sentence_boundary=False)
    (output, _) = preprocessor.run(documents=[document_1, document_2], id_hash_keys=['content', 'meta'])
    documents = output['documents']
    unique_ids = {d.id for d in documents}
    assert len(documents) == 4
    assert len(unique_ids) == 4

@pytest.mark.unit
@pytest.mark.parametrize('test_input', [(10, 0, True, 5), (10, 0, False, 4), (10, 5, True, 5), (10, 5, False, 7)])
def test_page_number_extraction(test_input):
    if False:
        print('Hello World!')
    (split_length, overlap, resp_sent_boundary, exp_doc_index) = test_input
    preprocessor = PreProcessor(add_page_number=True, split_by='word', split_length=split_length, split_overlap=overlap, split_respect_sentence_boundary=resp_sent_boundary)
    document = Document(content=TEXT)
    documents = preprocessor.process(document)
    for (idx, doc) in enumerate(documents):
        if idx < exp_doc_index:
            assert doc.meta['page'] == 1
        else:
            assert doc.meta['page'] == 2

@pytest.mark.unit
def test_page_number_extraction_on_empty_pages():
    if False:
        return 10
    '\n    Often "marketing" documents contain pages without text (visuals only). When extracting page numbers, these pages should be counted as well to avoid\n    issues when mapping results back to the original document.\n    '
    preprocessor = PreProcessor(add_page_number=True, split_by='word', split_length=7, split_overlap=0)
    text_page_one = 'This is a text on page one.'
    text_page_three = 'This is a text on page three.'
    document_with_empty_pages = f'{text_page_one}\x0c\x0c{text_page_three}'
    document = Document(content=document_with_empty_pages)
    documents = preprocessor.process(document)
    assert documents[0].meta['page'] == 1
    assert documents[1].meta['page'] == 3
    assert documents[0].content.strip() == text_page_one
    assert documents[1].content.strip() == text_page_three

@pytest.mark.unit
def test_headline_processing_split_by_word():
    if False:
        for i in range(10):
            print('nop')
    expected_headlines = [[{'headline': 'sample sentence in paragraph_1', 'start_idx': 11, 'level': 0}], [{'headline': 'sample sentence in paragraph_1', 'start_idx': None, 'level': 0}, {'headline': 'paragraph_1', 'start_idx': 19, 'level': 1}, {'headline': 'sample sentence in paragraph_2', 'start_idx': 44, 'level': 0}, {'headline': 'in paragraph_2', 'start_idx': 186, 'level': 1}], [{'headline': 'sample sentence in paragraph_2', 'start_idx': None, 'level': 0}, {'headline': 'in paragraph_2', 'start_idx': None, 'level': 1}, {'headline': 'sample sentence in paragraph_3', 'start_idx': 53, 'level': 0}], [{'headline': 'sample sentence in paragraph_3', 'start_idx': None, 'level': 0}, {'headline': 'trick the test', 'start_idx': 36, 'level': 1}]]
    document = Document(content=TEXT, meta={'headlines': HEADLINES})
    preprocessor = PreProcessor(split_length=30, split_overlap=0, split_by='word', split_respect_sentence_boundary=False)
    documents = preprocessor.process(document)
    for (doc, expected) in zip(documents, expected_headlines):
        assert doc.meta['headlines'] == expected

@pytest.mark.unit
def test_headline_processing_split_by_word_overlap():
    if False:
        return 10
    expected_headlines = [[{'headline': 'sample sentence in paragraph_1', 'start_idx': 11, 'level': 0}], [{'headline': 'sample sentence in paragraph_1', 'start_idx': None, 'level': 0}, {'headline': 'paragraph_1', 'start_idx': 71, 'level': 1}, {'headline': 'sample sentence in paragraph_2', 'start_idx': 96, 'level': 0}], [{'headline': 'sample sentence in paragraph_2', 'start_idx': None, 'level': 0}, {'headline': 'in paragraph_2', 'start_idx': 110, 'level': 1}, {'headline': 'sample sentence in paragraph_3', 'start_idx': 179, 'level': 0}], [{'headline': 'sample sentence in paragraph_2', 'start_idx': None, 'level': 0}, {'headline': 'in paragraph_2', 'start_idx': None, 'level': 1}, {'headline': 'sample sentence in paragraph_3', 'start_idx': 53, 'level': 0}], [{'headline': 'sample sentence in paragraph_3', 'start_idx': None, 'level': 0}, {'headline': 'trick the test', 'start_idx': 95, 'level': 1}]]
    document = Document(content=TEXT, meta={'headlines': HEADLINES})
    preprocessor = PreProcessor(split_length=30, split_overlap=10, split_by='word', split_respect_sentence_boundary=False)
    documents = preprocessor.process(document)
    for (doc, expected) in zip(documents, expected_headlines):
        assert doc.meta['headlines'] == expected

@pytest.mark.unit
def test_headline_processing_split_by_word_respect_sentence_boundary():
    if False:
        for i in range(10):
            print('nop')
    expected_headlines = [[{'headline': 'sample sentence in paragraph_1', 'start_idx': 11, 'level': 0}], [{'headline': 'sample sentence in paragraph_1', 'start_idx': None, 'level': 0}, {'headline': 'paragraph_1', 'start_idx': 71, 'level': 1}, {'headline': 'sample sentence in paragraph_2', 'start_idx': 96, 'level': 0}], [{'headline': 'sample sentence in paragraph_2', 'start_idx': None, 'level': 0}, {'headline': 'in paragraph_2', 'start_idx': 110, 'level': 1}], [{'headline': 'sample sentence in paragraph_2', 'start_idx': None, 'level': 0}, {'headline': 'in paragraph_2', 'start_idx': None, 'level': 1}, {'headline': 'sample sentence in paragraph_3', 'start_idx': 53, 'level': 0}], [{'headline': 'sample sentence in paragraph_3', 'start_idx': None, 'level': 0}, {'headline': 'trick the test', 'start_idx': 95, 'level': 1}]]
    document = Document(content=TEXT, meta={'headlines': HEADLINES})
    preprocessor = PreProcessor(split_length=30, split_overlap=5, split_by='word', split_respect_sentence_boundary=True)
    documents = preprocessor.process(document)
    for (doc, expected) in zip(documents, expected_headlines):
        assert doc.meta['headlines'] == expected

@pytest.mark.unit
def test_headline_processing_split_by_sentence():
    if False:
        return 10
    expected_headlines = [[{'headline': 'sample sentence in paragraph_1', 'start_idx': 11, 'level': 0}, {'headline': 'paragraph_1', 'start_idx': 198, 'level': 1}], [{'headline': 'sample sentence in paragraph_1', 'start_idx': None, 'level': 0}, {'headline': 'paragraph_1', 'start_idx': None, 'level': 1}, {'headline': 'sample sentence in paragraph_2', 'start_idx': 10, 'level': 0}, {'headline': 'in paragraph_2', 'start_idx': 152, 'level': 1}], [{'headline': 'sample sentence in paragraph_2', 'start_idx': None, 'level': 0}, {'headline': 'in paragraph_2', 'start_idx': None, 'level': 1}, {'headline': 'sample sentence in paragraph_3', 'start_idx': 10, 'level': 0}, {'headline': 'trick the test', 'start_idx': 179, 'level': 1}]]
    document = Document(content=TEXT, meta={'headlines': HEADLINES})
    preprocessor = PreProcessor(split_length=5, split_overlap=0, split_by='sentence', split_respect_sentence_boundary=False)
    documents = preprocessor.process(document)
    for (doc, expected) in zip(documents, expected_headlines):
        assert doc.meta['headlines'] == expected

@pytest.mark.unit
def test_headline_processing_split_by_sentence_overlap():
    if False:
        i = 10
        return i + 15
    expected_headlines = [[{'headline': 'sample sentence in paragraph_1', 'start_idx': 11, 'level': 0}, {'headline': 'paragraph_1', 'start_idx': 198, 'level': 1}], [{'headline': 'sample sentence in paragraph_1', 'start_idx': None, 'level': 0}, {'headline': 'paragraph_1', 'start_idx': 29, 'level': 1}, {'headline': 'sample sentence in paragraph_2', 'start_idx': 54, 'level': 0}, {'headline': 'in paragraph_2', 'start_idx': 196, 'level': 1}], [{'headline': 'sample sentence in paragraph_2', 'start_idx': None, 'level': 0}, {'headline': 'in paragraph_2', 'start_idx': 26, 'level': 1}, {'headline': 'sample sentence in paragraph_3', 'start_idx': 95, 'level': 0}], [{'headline': 'sample sentence in paragraph_3', 'start_idx': None, 'level': 0}, {'headline': 'trick the test', 'start_idx': 95, 'level': 1}]]
    document = Document(content=TEXT, meta={'headlines': HEADLINES})
    preprocessor = PreProcessor(split_length=5, split_overlap=1, split_by='sentence', split_respect_sentence_boundary=False)
    documents = preprocessor.process(document)
    for (doc, expected) in zip(documents, expected_headlines):
        assert doc.meta['headlines'] == expected

@pytest.mark.unit
def test_headline_processing_split_by_passage():
    if False:
        i = 10
        return i + 15
    expected_headlines = [[{'headline': 'sample sentence in paragraph_1', 'start_idx': 11, 'level': 0}, {'headline': 'paragraph_1', 'start_idx': 198, 'level': 1}], [{'headline': 'sample sentence in paragraph_1', 'start_idx': None, 'level': 0}, {'headline': 'paragraph_1', 'start_idx': None, 'level': 1}, {'headline': 'sample sentence in paragraph_2', 'start_idx': 10, 'level': 0}, {'headline': 'in paragraph_2', 'start_idx': 152, 'level': 1}], [{'headline': 'sample sentence in paragraph_2', 'start_idx': None, 'level': 0}, {'headline': 'in paragraph_2', 'start_idx': None, 'level': 1}, {'headline': 'sample sentence in paragraph_3', 'start_idx': 10, 'level': 0}, {'headline': 'trick the test', 'start_idx': 179, 'level': 1}]]
    document = Document(content=TEXT, meta={'headlines': HEADLINES})
    preprocessor = PreProcessor(split_length=1, split_overlap=0, split_by='passage', split_respect_sentence_boundary=False)
    documents = preprocessor.process(document)
    for (doc, expected) in zip(documents, expected_headlines):
        assert doc.meta['headlines'] == expected

@pytest.mark.unit
def test_headline_processing_split_by_passage_overlap():
    if False:
        i = 10
        return i + 15
    expected_headlines = [[{'headline': 'sample sentence in paragraph_1', 'start_idx': 11, 'level': 0}, {'headline': 'paragraph_1', 'start_idx': 198, 'level': 1}, {'headline': 'sample sentence in paragraph_2', 'start_idx': 223, 'level': 0}, {'headline': 'in paragraph_2', 'start_idx': 365, 'level': 1}], [{'headline': 'sample sentence in paragraph_1', 'start_idx': None, 'level': 0}, {'headline': 'paragraph_1', 'start_idx': None, 'level': 1}, {'headline': 'sample sentence in paragraph_2', 'start_idx': 10, 'level': 0}, {'headline': 'in paragraph_2', 'start_idx': 152, 'level': 1}, {'headline': 'sample sentence in paragraph_3', 'start_idx': 221, 'level': 0}, {'headline': 'trick the test', 'start_idx': 390, 'level': 1}]]
    document = Document(content=TEXT, meta={'headlines': HEADLINES})
    preprocessor = PreProcessor(split_length=2, split_overlap=1, split_by='passage', split_respect_sentence_boundary=False)
    documents = preprocessor.process(document)
    for (doc, expected) in zip(documents, expected_headlines):
        assert doc.meta['headlines'] == expected

@pytest.mark.unit
def test_file_exists_error_during_download(monkeypatch: MonkeyPatch, module_tmp_dir: Path):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setattr(nltk.data, 'find', Mock(side_effect=[LookupError, str(module_tmp_dir)]))
    monkeypatch.setattr(nltk, 'download', Mock(side_effect=FileExistsError))
    PreProcessor(split_length=2, split_respect_sentence_boundary=False)

@pytest.mark.unit
def test_preprocessor_very_long_document(caplog):
    if False:
        while True:
            i = 10
    preproc = PreProcessor(clean_empty_lines=False, clean_header_footer=False, clean_whitespace=False, split_by=None, max_chars_check=10)
    documents = [Document(content=str(i) + '.' * i) for i in range(0, 30, 3)]
    results = preproc.process(documents)
    assert len(results) == 19
    assert any((d.content.startswith('.') for d in results))
    assert any((not d.content.startswith('.') for d in results))
    assert 'characters long after preprocessing, where the maximum length should be 10.' in caplog.text

@pytest.mark.unit
def test_split_respect_sentence_boundary_exceeding_split_len_not_repeated():
    if False:
        while True:
            i = 10
    preproc = PreProcessor(split_length=13, split_overlap=3, split_by='word', split_respect_sentence_boundary=True)
    document = Document(content='This is a test sentence with many many words that exceeds the split length and should not be repeated. This is another test sentence. (This is a third test sentence.) This is the last test sentence.')
    documents = preproc.process(document)
    assert len(documents) == 3
    assert documents[0].content == 'This is a test sentence with many many words that exceeds the split length and should not be repeated. '
    assert 'This is a test sentence with many many words' not in documents[1].content
    assert 'This is a test sentence with many many words' not in documents[2].content

@pytest.mark.unit
def test_split_overlap_information():
    if False:
        for i in range(10):
            print('nop')
    preproc = PreProcessor(split_length=13, split_overlap=3, split_by='word', split_respect_sentence_boundary=True)
    document = Document(content='This is a test sentence with many many words that exceeds the split length and should not be repeated. This is another test sentence. (This is a third test sentence.) This is the fourth sentence. This is the last test sentence.')
    documents = preproc.process(document)
    assert len(documents) == 4
    assert len(documents[0].meta['_split_overlap']) == 0
    assert len(documents[1].meta['_split_overlap']) == 1
    assert len(documents[2].meta['_split_overlap']) == 2
    assert len(documents[3].meta['_split_overlap']) == 1
    assert documents[1].meta['_split_overlap'][0]['doc_id'] == documents[2].id
    assert documents[2].meta['_split_overlap'][0]['doc_id'] == documents[1].id
    assert documents[2].meta['_split_overlap'][1]['doc_id'] == documents[3].id
    assert documents[3].meta['_split_overlap'][0]['doc_id'] == documents[2].id
    doc1_overlap_doc2 = documents[1].meta['_split_overlap'][0]['range']
    doc2_overlap_doc1 = documents[2].meta['_split_overlap'][0]['range']
    assert documents[1].content[doc1_overlap_doc2[0]:doc1_overlap_doc2[1]] == documents[2].content[doc2_overlap_doc1[0]:doc2_overlap_doc1[1]]
    doc2_overlap_doc3 = documents[2].meta['_split_overlap'][1]['range']
    doc3_overlap_doc2 = documents[3].meta['_split_overlap'][0]['range']
    assert documents[2].content[doc2_overlap_doc3[0]:doc2_overlap_doc3[1]] == documents[3].content[doc3_overlap_doc2[0]:doc3_overlap_doc2[1]]