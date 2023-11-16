from datetime import datetime
import logging
import pytest
import haystack
from haystack import Pipeline, Document, Answer
from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.nodes.other.shaper import Shaper
from haystack.nodes.retriever.sparse import BM25Retriever

@pytest.fixture
def mock_function(monkeypatch):
    if False:
        return 10
    monkeypatch.setattr(haystack.nodes.other.shaper, 'REGISTERED_FUNCTIONS', {'test_function': lambda a, b: [a] * len(b)})

@pytest.fixture
def mock_function_two_outputs(monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setattr(haystack.nodes.other.shaper, 'REGISTERED_FUNCTIONS', {'two_output_test_function': lambda a: (a, len(a))})

@pytest.mark.unit
def test_basic_invocation_only_inputs(mock_function):
    if False:
        print('Hello World!')
    shaper = Shaper(func='test_function', inputs={'a': 'query', 'b': 'documents'}, outputs=['c'])
    (results, _) = shaper.run(query='test query', documents=["doesn't", 'really', 'matter'])
    assert results['invocation_context']['c'] == ['test query', 'test query', 'test query']

@pytest.mark.unit
def test_basic_invocation_empty_documents_list(mock_function):
    if False:
        for i in range(10):
            print('nop')
    shaper = Shaper(func='test_function', inputs={'a': 'query', 'b': 'documents'}, outputs=['c'])
    (results, _) = shaper.run(query='test query', documents=[])
    assert results['invocation_context']['c'] == []

@pytest.mark.unit
def test_multiple_outputs(mock_function_two_outputs):
    if False:
        print('Hello World!')
    shaper = Shaper(func='two_output_test_function', inputs={'a': 'query'}, outputs=['c', 'd'])
    (results, _) = shaper.run(query='test')
    assert results['invocation_context']['c'] == 'test'
    assert results['invocation_context']['d'] == 4

@pytest.mark.unit
def test_multiple_outputs_error(mock_function_two_outputs, caplog):
    if False:
        print('Hello World!')
    shaper = Shaper(func='two_output_test_function', inputs={'a': 'query'}, outputs=['c'])
    with caplog.at_level(logging.WARNING):
        (results, _) = shaper.run(query='test')
        assert 'Only 1 output(s) will be stored.' in caplog.text

@pytest.mark.unit
def test_basic_invocation_only_params(mock_function):
    if False:
        while True:
            i = 10
    shaper = Shaper(func='test_function', params={'a': 'A', 'b': list(range(3))}, outputs=['c'])
    (results, _) = shaper.run()
    assert results['invocation_context']['c'] == ['A', 'A', 'A']

@pytest.mark.unit
def test_basic_invocation_inputs_and_params(mock_function):
    if False:
        for i in range(10):
            print('nop')
    shaper = Shaper(func='test_function', inputs={'a': 'query'}, params={'b': list(range(2))}, outputs=['c'])
    (results, _) = shaper.run(query='test query')
    assert results['invocation_context']['c'] == ['test query', 'test query']

@pytest.mark.unit
def test_basic_invocation_inputs_and_params_colliding(mock_function):
    if False:
        while True:
            i = 10
    shaper = Shaper(func='test_function', inputs={'a': 'query'}, params={'a': 'default value', 'b': list(range(2))}, outputs=['c'])
    (results, _) = shaper.run(query='test query')
    assert results['invocation_context']['c'] == ['test query', 'test query']

@pytest.mark.unit
def test_basic_invocation_inputs_and_params_using_params_as_defaults(mock_function):
    if False:
        i = 10
        return i + 15
    shaper = Shaper(func='test_function', inputs={'a': 'query'}, params={'a': 'default', 'b': list(range(2))}, outputs=['c'])
    (results, _) = shaper.run()
    assert results['invocation_context']['c'] == ['default', 'default']

@pytest.mark.unit
def test_missing_argument(mock_function):
    if False:
        return 10
    shaper = Shaper(func='test_function', inputs={'b': 'documents'}, outputs=['c'])
    with pytest.raises(ValueError, match="Shaper couldn't apply the function to your inputs and parameters."):
        shaper.run(query='test query', documents=["doesn't", 'really', 'matter'])

@pytest.mark.unit
def test_excess_argument(mock_function):
    if False:
        print('Hello World!')
    shaper = Shaper(func='test_function', inputs={'a': 'query', 'b': 'documents', 'something_extra': 'query'}, outputs=['c'])
    with pytest.raises(ValueError, match="Shaper couldn't apply the function to your inputs and parameters."):
        shaper.run(query='test query', documents=["doesn't", 'really', 'matter'])

@pytest.mark.unit
def test_value_not_in_invocation_context(mock_function):
    if False:
        for i in range(10):
            print('nop')
    shaper = Shaper(func='test_function', inputs={'a': 'query', 'b': 'something_that_does_not_exist'}, outputs=['c'])
    with pytest.raises(ValueError, match="Shaper couldn't apply the function to your inputs and parameters."):
        shaper.run(query='test query', documents=["doesn't", 'really', 'matter'])

@pytest.mark.unit
def test_value_only_in_invocation_context(mock_function):
    if False:
        for i in range(10):
            print('nop')
    shaper = Shaper(func='test_function', inputs={'a': 'query', 'b': 'invocation_context_specific'}, outputs=['c'])
    (results, _s) = shaper.run(query='test query', invocation_context={'invocation_context_specific': ["doesn't", 'really', 'matter']})
    assert results['invocation_context']['c'] == ['test query', 'test query', 'test query']

@pytest.mark.unit
def test_yaml(mock_function, tmp_path):
    if False:
        i = 10
        return i + 15
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: shaper\n              type: Shaper\n              params:\n                func: test_function\n                inputs:\n                  a: query\n                params:\n                  b: [1, 1]\n                outputs:\n                  - c\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper\n                    inputs:\n                      - Query\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(query='test query', documents=[Document(content='first'), Document(content='second'), Document(content='third')])
    assert result['invocation_context']['c'] == ['test query', 'test query']
    assert result['query'] == 'test query'
    assert result['documents'] == [Document(content='first'), Document(content='second'), Document(content='third')]

@pytest.mark.unit
def test_rename():
    if False:
        for i in range(10):
            print('nop')
    shaper = Shaper(func='rename', inputs={'value': 'query'}, outputs=['questions'])
    (results, _) = shaper.run(query='test query')
    assert results['invocation_context']['questions'] == 'test query'

@pytest.mark.unit
def test_rename_yaml(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: shaper\n              type: Shaper\n              params:\n                func: rename\n                inputs:\n                  value: query\n                outputs:\n                  - questions\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper\n                    inputs:\n                      - Query\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(query='test query')
    assert result['invocation_context']['query'] == 'test query'
    assert result['invocation_context']['questions'] == 'test query'

@pytest.mark.unit
def test_current_datetime():
    if False:
        while True:
            i = 10
    shaper = Shaper(func='current_datetime', inputs={}, outputs=['date_time'], params={'format': '%y-%m-%d'})
    (results, _) = shaper.run()
    assert results['invocation_context']['date_time'] == datetime.now().strftime('%y-%m-%d')

@pytest.mark.unit
def test_current_datetime_yaml(tmp_path):
    if False:
        print('Hello World!')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: shaper\n              type: Shaper\n              params:\n                func: current_datetime\n                params:\n                  format: "%y-%m-%d"\n                outputs:\n                  - date_time\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper\n                    inputs:\n                      - Query\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run()
    assert result['invocation_context']['date_time'] == datetime.now().strftime('%y-%m-%d')

@pytest.mark.unit
def test_value_to_list():
    if False:
        i = 10
        return i + 15
    shaper = Shaper(func='value_to_list', inputs={'value': 'query', 'target_list': 'documents'}, outputs=['questions'])
    (results, _) = shaper.run(query='test query', documents=["doesn't", 'really', 'matter'])
    assert results['invocation_context']['questions'] == ['test query', 'test query', 'test query']

@pytest.mark.unit
def test_value_to_list_yaml(tmp_path):
    if False:
        print('Hello World!')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: shaper\n              type: Shaper\n              params:\n                func: value_to_list\n                inputs:\n                  value: query\n                  target_list: documents\n                outputs:\n                  - questions\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper\n                    inputs:\n                      - Query\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(query='test query', documents=[Document(content='first'), Document(content='second'), Document(content='third')])
    assert result['invocation_context']['questions'] == ['test query', 'test query', 'test query']
    assert result['query'] == 'test query'
    assert result['documents'] == [Document(content='first'), Document(content='second'), Document(content='third')]

@pytest.mark.unit
def test_join_lists():
    if False:
        return 10
    shaper = Shaper(func='join_lists', params={'lists': [[1, 2, 3], [4, 5]]}, outputs=['list'])
    (results, _) = shaper.run()
    assert results['invocation_context']['list'] == [1, 2, 3, 4, 5]

@pytest.mark.unit
def test_join_lists_yaml(tmp_path):
    if False:
        i = 10
        return i + 15
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: shaper\n              type: Shaper\n              params:\n                func: join_lists\n                inputs:\n                  lists:\n                   - documents\n                   - file_paths\n                outputs:\n                  - single_list\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper\n                    inputs:\n                      - Query\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(documents=['first', 'second', 'third'], file_paths=['file1.txt', 'file2.txt'])
    assert result['invocation_context']['single_list'] == ['first', 'second', 'third', 'file1.txt', 'file2.txt']

@pytest.mark.unit
def test_join_strings():
    if False:
        i = 10
        return i + 15
    shaper = Shaper(func='join_strings', params={'strings': ['first', 'second'], 'delimiter': ' | '}, outputs=['single_string'])
    (results, _) = shaper.run()
    assert results['invocation_context']['single_string'] == 'first | second'

@pytest.mark.unit
def test_join_strings_default_delimiter():
    if False:
        i = 10
        return i + 15
    shaper = Shaper(func='join_strings', params={'strings': ['first', 'second']}, outputs=['single_string'])
    (results, _) = shaper.run()
    assert results['invocation_context']['single_string'] == 'first second'

@pytest.mark.unit
def test_join_strings_with_str_replace():
    if False:
        for i in range(10):
            print('nop')
    shaper = Shaper(func='join_strings', params={'strings': ['first', 'second', 'third'], 'delimiter': ' - ', 'str_replace': {'r': 'R'}}, outputs=['single_string'])
    (results, _) = shaper.run()
    assert results['invocation_context']['single_string'] == 'fiRst - second - thiRd'

@pytest.mark.unit
def test_join_strings_yaml(tmp_path):
    if False:
        i = 10
        return i + 15
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write("\n            version: ignore\n            components:\n            - name: shaper\n              type: Shaper\n              params:\n                func: join_strings\n                inputs:\n                  strings: documents\n                params:\n                  delimiter: ' - '\n                outputs:\n                  - single_string\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper\n                    inputs:\n                      - Query\n        ")
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(documents=['first', 'second', 'third'])
    assert result['invocation_context']['single_string'] == 'first - second - third'

@pytest.mark.unit
def test_join_strings_default_delimiter_yaml(tmp_path):
    if False:
        print('Hello World!')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: shaper\n              type: Shaper\n              params:\n                func: join_strings\n                inputs:\n                  strings: documents\n                outputs:\n                  - single_string\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper\n                    inputs:\n                      - Query\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(documents=['first', 'second', 'third'])
    assert result['invocation_context']['single_string'] == 'first second third'

@pytest.mark.unit
def test_join_strings_with_str_replace_yaml(tmp_path):
    if False:
        i = 10
        return i + 15
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write("\n            version: ignore\n            components:\n            - name: shaper\n              type: Shaper\n              params:\n                func: join_strings\n                inputs:\n                  strings: documents\n                outputs:\n                  - single_string\n                params:\n                  delimiter: ' - '\n                  str_replace:\n                    r: R\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper\n                    inputs:\n                      - Query\n        ")
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(documents=['first', 'second', 'third'])
    assert result['invocation_context']['single_string'] == 'fiRst - second - thiRd'

@pytest.mark.unit
def test_join_documents():
    if False:
        i = 10
        return i + 15
    shaper = Shaper(func='join_documents', inputs={'documents': 'documents'}, params={'delimiter': ' | '}, outputs=['documents'])
    (results, _) = shaper.run(documents=[Document(content='first'), Document(content='second'), Document(content='third')])
    assert results['invocation_context']['documents'] == [Document(content='first | second | third')]
    assert results['documents'] == [Document(content='first | second | third')]

def test_join_documents_without_publish_outputs():
    if False:
        i = 10
        return i + 15
    shaper = Shaper(func='join_documents', inputs={'documents': 'documents'}, params={'delimiter': ' | '}, outputs=['documents'], publish_outputs=False)
    (results, _) = shaper.run(documents=[Document(content='first'), Document(content='second'), Document(content='third')])
    assert results['invocation_context']['documents'] == [Document(content='first | second | third')]
    assert 'documents' not in results

def test_join_documents_with_publish_outputs_as_list():
    if False:
        i = 10
        return i + 15
    shaper = Shaper(func='join_documents', inputs={'documents': 'documents'}, params={'delimiter': ' | '}, outputs=['documents'], publish_outputs=['documents'])
    (results, _) = shaper.run(documents=[Document(content='first'), Document(content='second'), Document(content='third')])
    assert results['invocation_context']['documents'] == [Document(content='first | second | third')]
    assert results['documents'] == [Document(content='first | second | third')]

@pytest.mark.unit
def test_join_documents_default_delimiter():
    if False:
        while True:
            i = 10
    shaper = Shaper(func='join_documents', inputs={'documents': 'documents'}, outputs=['documents'])
    (results, _) = shaper.run(documents=[Document(content='first'), Document(content='second'), Document(content='third')])
    assert results['invocation_context']['documents'] == [Document(content='first second third')]

@pytest.mark.unit
def test_join_documents_with_pattern_and_str_replace():
    if False:
        print('Hello World!')
    shaper = Shaper(func='join_documents', inputs={'documents': 'documents'}, outputs=['documents'], params={'delimiter': ' - ', 'pattern': '[$idx] $content', 'str_replace': {'r': 'R'}})
    (results, _) = shaper.run(documents=[Document(content='first'), Document(content='second'), Document(content='third')])
    assert results['invocation_context']['documents'] == [Document(content='[1] fiRst - [2] second - [3] thiRd')]

@pytest.mark.unit
def test_join_documents_yaml(tmp_path):
    if False:
        print('Hello World!')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write("\n            version: ignore\n\n            components:\n            - name: shaper\n              type: Shaper\n              params:\n                func: join_documents\n                inputs:\n                  documents: documents\n                params:\n                  delimiter: ' - '\n                outputs:\n                  - documents\n\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper\n                    inputs:\n                      - Query\n        ")
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(query='test query', documents=[Document(content='first'), Document(content='second'), Document(content='third')])
    assert result['invocation_context']['documents'] == [Document(content='first - second - third')]
    assert result['documents'] == [Document(content='first - second - third')]

@pytest.mark.unit
def test_join_documents_default_delimiter_yaml(tmp_path):
    if False:
        print('Hello World!')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: shaper\n              type: Shaper\n              params:\n                func: join_documents\n                inputs:\n                  documents: documents\n                outputs:\n                  - documents\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper\n                    inputs:\n                      - Query\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(query='test query', documents=[Document(content='first'), Document(content='second'), Document(content='third')])
    assert result['invocation_context']['documents'] == [Document(content='first second third')]

@pytest.mark.unit
def test_join_documents_with_pattern_and_str_replace_yaml(tmp_path):
    if False:
        print('Hello World!')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write("\n            version: ignore\n            components:\n            - name: shaper\n              type: Shaper\n              params:\n                func: join_documents\n                inputs:\n                  documents: documents\n                outputs:\n                  - documents\n                params:\n                  delimiter: ' - '\n                  pattern: '[$idx] $content'\n                  str_replace:\n                    r: R\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper\n                    inputs:\n                      - Query\n        ")
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(query='test query', documents=[Document(content='first'), Document(content='second'), Document(content='third')])
    assert result['invocation_context']['documents'] == [Document(content='[1] fiRst - [2] second - [3] thiRd')]

@pytest.mark.unit
def test_strings_to_answers_simple():
    if False:
        i = 10
        return i + 15
    shaper = Shaper(func='strings_to_answers', inputs={'strings': 'responses'}, outputs=['answers'])
    (results, _) = shaper.run(invocation_context={'responses': ['first', 'second', 'third']})
    assert results['invocation_context']['answers'] == [Answer(answer='first', type='generative', meta={'prompt': None}), Answer(answer='second', type='generative', meta={'prompt': None}), Answer(answer='third', type='generative', meta={'prompt': None})]

@pytest.mark.unit
def test_strings_to_answers_with_prompt():
    if False:
        for i in range(10):
            print('nop')
    shaper = Shaper(func='strings_to_answers', inputs={'strings': 'responses'}, outputs=['answers'])
    (results, _) = shaper.run(invocation_context={'responses': ['first', 'second', 'third'], 'prompts': ['test prompt']})
    assert results['invocation_context']['answers'] == [Answer(answer='first', type='generative', meta={'prompt': 'test prompt'}), Answer(answer='second', type='generative', meta={'prompt': 'test prompt'}), Answer(answer='third', type='generative', meta={'prompt': 'test prompt'})]

@pytest.mark.unit
def test_strings_to_answers_with_documents():
    if False:
        while True:
            i = 10
    shaper = Shaper(func='strings_to_answers', inputs={'strings': 'responses'}, outputs=['answers'])
    (results, _) = shaper.run(invocation_context={'responses': ['first', 'second', 'third'], 'documents': [Document(id='123', content='test'), Document(id='456', content='test')]})
    assert results['invocation_context']['answers'] == [Answer(answer='first', type='generative', meta={'prompt': None}, document_ids=['123', '456']), Answer(answer='second', type='generative', meta={'prompt': None}, document_ids=['123', '456']), Answer(answer='third', type='generative', meta={'prompt': None}, document_ids=['123', '456'])]

@pytest.mark.unit
def test_strings_to_answers_with_prompt_per_document():
    if False:
        print('Hello World!')
    shaper = Shaper(func='strings_to_answers', inputs={'strings': 'responses'}, outputs=['answers'])
    (results, _) = shaper.run(invocation_context={'responses': ['first', 'second'], 'documents': [Document(id='123', content='test'), Document(id='456', content='test')], 'prompts': ['prompt1', 'prompt2']})
    assert results['invocation_context']['answers'] == [Answer(answer='first', type='generative', meta={'prompt': 'prompt1'}, document_ids=['123']), Answer(answer='second', type='generative', meta={'prompt': 'prompt2'}, document_ids=['456'])]

@pytest.mark.unit
def test_strings_to_answers_with_prompt_per_document_multiple_results():
    if False:
        return 10
    shaper = Shaper(func='strings_to_answers', inputs={'strings': 'responses'}, outputs=['answers'])
    (results, _) = shaper.run(invocation_context={'responses': ['first', 'second', 'third', 'fourth'], 'documents': [Document(id='123', content='test'), Document(id='456', content='test')], 'prompts': ['prompt1', 'prompt2']})
    assert results['invocation_context']['answers'] == [Answer(answer='first', type='generative', meta={'prompt': 'prompt1'}, document_ids=['123']), Answer(answer='second', type='generative', meta={'prompt': 'prompt1'}, document_ids=['123']), Answer(answer='third', type='generative', meta={'prompt': 'prompt2'}, document_ids=['456']), Answer(answer='fourth', type='generative', meta={'prompt': 'prompt2'}, document_ids=['456'])]

@pytest.mark.unit
def test_strings_to_answers_with_pattern_group():
    if False:
        i = 10
        return i + 15
    shaper = Shaper(func='strings_to_answers', inputs={'strings': 'responses'}, outputs=['answers'], params={'pattern': 'Answer: (.*)'})
    (results, _) = shaper.run(invocation_context={'responses': ['Answer: first', 'Answer: second', 'Answer: third']})
    assert results['invocation_context']['answers'] == [Answer(answer='first', type='generative', meta={'prompt': None}), Answer(answer='second', type='generative', meta={'prompt': None}), Answer(answer='third', type='generative', meta={'prompt': None})]

@pytest.mark.unit
def test_strings_to_answers_with_pattern_no_group():
    if False:
        i = 10
        return i + 15
    shaper = Shaper(func='strings_to_answers', inputs={'strings': 'responses'}, outputs=['answers'], params={'pattern': '[^\\n]+$'})
    (results, _) = shaper.run(invocation_context={'responses': ['Answer\nfirst', 'Answer\nsecond', 'Answer\n\nthird']})
    assert results['invocation_context']['answers'] == [Answer(answer='first', type='generative', meta={'prompt': None}), Answer(answer='second', type='generative', meta={'prompt': None}), Answer(answer='third', type='generative', meta={'prompt': None})]

@pytest.mark.unit
def test_strings_to_answers_with_references_index():
    if False:
        print('Hello World!')
    shaper = Shaper(func='strings_to_answers', inputs={'strings': 'responses', 'documents': 'documents'}, outputs=['answers'], params={'reference_pattern': '\\[(\\d+)\\]'})
    (results, _) = shaper.run(invocation_context={'responses': ['first[1]', 'second[2]', 'third[1][2]', 'fourth'], 'documents': [Document(id='123', content='test'), Document(id='456', content='test')]})
    assert results['invocation_context']['answers'] == [Answer(answer='first[1]', type='generative', meta={'prompt': None}, document_ids=['123']), Answer(answer='second[2]', type='generative', meta={'prompt': None}, document_ids=['456']), Answer(answer='third[1][2]', type='generative', meta={'prompt': None}, document_ids=['123', '456']), Answer(answer='fourth', type='generative', meta={'prompt': None}, document_ids=[])]

@pytest.mark.unit
def test_strings_to_answers_with_references_id():
    if False:
        i = 10
        return i + 15
    shaper = Shaper(func='strings_to_answers', inputs={'strings': 'responses', 'documents': 'documents'}, outputs=['answers'], params={'reference_pattern': '\\[(\\d+)\\]', 'reference_mode': 'id'})
    (results, _) = shaper.run(invocation_context={'responses': ['first[123]', 'second[456]', 'third[123][456]', 'fourth'], 'documents': [Document(id='123', content='test'), Document(id='456', content='test')]})
    assert results['invocation_context']['answers'] == [Answer(answer='first[123]', type='generative', meta={'prompt': None}, document_ids=['123']), Answer(answer='second[456]', type='generative', meta={'prompt': None}, document_ids=['456']), Answer(answer='third[123][456]', type='generative', meta={'prompt': None}, document_ids=['123', '456']), Answer(answer='fourth', type='generative', meta={'prompt': None}, document_ids=[])]

@pytest.mark.unit
def test_strings_to_answers_with_references_meta():
    if False:
        print('Hello World!')
    shaper = Shaper(func='strings_to_answers', inputs={'strings': 'responses', 'documents': 'documents'}, outputs=['answers'], params={'reference_pattern': '\\[([^\\]]+)\\]', 'reference_mode': 'meta', 'reference_meta_field': 'file_id'})
    (results, _) = shaper.run(invocation_context={'responses': ['first[123.txt]', 'second[456.txt]', 'third[123.txt][456.txt]', 'fourth'], 'documents': [Document(id='123', content='test', meta={'file_id': '123.txt'}), Document(id='456', content='test', meta={'file_id': '456.txt'})]})
    assert results['invocation_context']['answers'] == [Answer(answer='first[123.txt]', type='generative', meta={'prompt': None}, document_ids=['123']), Answer(answer='second[456.txt]', type='generative', meta={'prompt': None}, document_ids=['456']), Answer(answer='third[123.txt][456.txt]', type='generative', meta={'prompt': None}, document_ids=['123', '456']), Answer(answer='fourth', type='generative', meta={'prompt': None}, document_ids=[])]

@pytest.mark.unit
def test_strings_to_answers_yaml(tmp_path):
    if False:
        return 10
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write("\n            version: ignore\n            components:\n            - name: shaper\n              type: Shaper\n              params:\n                func: strings_to_answers\n                params:\n                  strings: ['a', 'b', 'c']\n                outputs:\n                  - answers\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper\n                    inputs:\n                      - Query\n        ")
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run()
    assert result['invocation_context']['answers'] == [Answer(answer='a', type='generative', meta={'prompt': None}), Answer(answer='b', type='generative', meta={'prompt': None}), Answer(answer='c', type='generative', meta={'prompt': None})]
    assert result['answers'] == [Answer(answer='a', type='generative', meta={'prompt': None}), Answer(answer='b', type='generative', meta={'prompt': None}), Answer(answer='c', type='generative', meta={'prompt': None})]

@pytest.mark.unit
def test_strings_to_answers_with_reference_meta_yaml(tmp_path):
    if False:
        while True:
            i = 10
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write("\n            version: ignore\n            components:\n            - name: shaper\n              type: Shaper\n              params:\n                func: strings_to_answers\n                inputs:\n                  documents: documents\n                params:\n                  reference_meta_field: file_id\n                  reference_mode: meta\n                  reference_pattern: \\[([^\\]]+)\\]\n                  strings: ['first[123.txt]', 'second[456.txt]', 'third[123.txt][456.txt]', 'fourth']\n                outputs:\n                  - answers\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper\n                    inputs:\n                      - Query\n        ")
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(documents=[Document(id='123', content='test', meta={'file_id': '123.txt'}), Document(id='456', content='test', meta={'file_id': '456.txt'})])
    assert result['invocation_context']['answers'] == [Answer(answer='first[123.txt]', type='generative', meta={'prompt': None}, document_ids=['123']), Answer(answer='second[456.txt]', type='generative', meta={'prompt': None}, document_ids=['456']), Answer(answer='third[123.txt][456.txt]', type='generative', meta={'prompt': None}, document_ids=['123', '456']), Answer(answer='fourth', type='generative', meta={'prompt': None}, document_ids=[])]
    assert result['answers'] == [Answer(answer='first[123.txt]', type='generative', meta={'prompt': None}, document_ids=['123']), Answer(answer='second[456.txt]', type='generative', meta={'prompt': None}, document_ids=['456']), Answer(answer='third[123.txt][456.txt]', type='generative', meta={'prompt': None}, document_ids=['123', '456']), Answer(answer='fourth', type='generative', meta={'prompt': None}, document_ids=[])]

@pytest.mark.integration
def test_strings_to_answers_after_prompt_node_yaml(tmp_path):
    if False:
        print('Hello World!')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write("\n            version: ignore\n            components:\n              - name: prompt_model\n                type: PromptModel\n\n              - name: prompt_template_raw_qa_per_document\n                type: PromptTemplate\n                params:\n                  prompt: 'Given the context please answer the question. Context: {documents}; Question: {query}; Answer:'\n\n              - name: prompt_node_raw_qa\n                type: PromptNode\n                params:\n                  model_name_or_path: prompt_model\n                  default_prompt_template: prompt_template_raw_qa_per_document\n                  top_k: 2\n\n              - name: prompt_node_question_generation\n                type: PromptNode\n                params:\n                  model_name_or_path: prompt_model\n                  default_prompt_template: question-generation\n                  output_variable: query\n\n              - name: shaper\n                type: Shaper\n                params:\n                  func: strings_to_answers\n                  inputs:\n                    strings: results\n                  outputs:\n                    - answers\n\n\n            pipelines:\n              - name: query\n                nodes:\n                  - name: prompt_node_question_generation\n                    inputs:\n                      - Query\n                  - name: prompt_node_raw_qa\n                    inputs:\n                      - prompt_node_question_generation\n                  - name: shaper\n                    inputs:\n                      - prompt_node_raw_qa\n            ")
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(query="What's Berlin like?", documents=[Document('Berlin is an amazing city.', id='123'), Document('Berlin is a cool city in Germany.', id='456')])
    results = result['answers']
    assert len(results) == 4
    assert any((True for r in results if 'Berlin' in r.answer))
    for answer in results[:2]:
        assert answer.document_ids == ['123']
        assert answer.meta['prompt'] == f"Given the context please answer the question. Context: Berlin is an amazing city.; Question: {result['query'][0]}; Answer:"
    for answer in results[2:]:
        assert answer.document_ids == ['456']
        assert answer.meta['prompt'] == f"Given the context please answer the question. Context: Berlin is a cool city in Germany.; Question: {result['query'][1]}; Answer:"

@pytest.mark.unit
def test_answers_to_strings():
    if False:
        while True:
            i = 10
    shaper = Shaper(func='answers_to_strings', inputs={'answers': 'documents'}, outputs=['strings'])
    (results, _) = shaper.run(documents=[Answer(answer='first'), Answer(answer='second'), Answer(answer='third')])
    assert results['invocation_context']['strings'] == ['first', 'second', 'third']

@pytest.mark.unit
def test_answers_to_strings_with_pattern_and_str_replace():
    if False:
        return 10
    shaper = Shaper(func='answers_to_strings', inputs={'answers': 'documents'}, outputs=['strings'], params={'pattern': '[$idx] $answer', 'str_replace': {'r': 'R'}})
    (results, _) = shaper.run(documents=[Answer(answer='first'), Answer(answer='second'), Answer(answer='third')])
    assert results['invocation_context']['strings'] == ['[1] fiRst', '[2] second', '[3] thiRd']

@pytest.mark.unit
def test_answers_to_strings_yaml(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: shaper\n              type: Shaper\n              params:\n                func: answers_to_strings\n                inputs:\n                  answers: documents\n                outputs:\n                  - strings\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper\n                    inputs:\n                      - Query\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(documents=[Answer(answer='a'), Answer(answer='b'), Answer(answer='c')])
    assert result['invocation_context']['strings'] == ['a', 'b', 'c']

@pytest.mark.unit
def test_answers_to_strings_with_pattern_and_str_yaml(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write("\n            version: ignore\n            components:\n            - name: shaper\n              type: Shaper\n              params:\n                func: answers_to_strings\n                inputs:\n                  answers: documents\n                outputs:\n                  - strings\n                params:\n                  pattern: '[$idx] $answer'\n                  str_replace:\n                    r: R\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper\n                    inputs:\n                      - Query\n        ")
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(documents=[Answer(answer='first'), Answer(answer='second'), Answer(answer='third')])
    assert result['invocation_context']['strings'] == ['[1] fiRst', '[2] second', '[3] thiRd']

@pytest.mark.unit
def test_strings_to_documents_no_meta_no_hashkeys():
    if False:
        return 10
    shaper = Shaper(func='strings_to_documents', inputs={'strings': 'responses'}, outputs=['documents'])
    (results, _) = shaper.run(invocation_context={'responses': ['first', 'second', 'third']})
    assert results['invocation_context']['documents'] == [Document(content='first'), Document(content='second'), Document(content='third')]

@pytest.mark.unit
def test_strings_to_documents_single_meta_no_hashkeys():
    if False:
        i = 10
        return i + 15
    shaper = Shaper(func='strings_to_documents', inputs={'strings': 'responses'}, params={'meta': {'a': 'A'}}, outputs=['documents'])
    (results, _) = shaper.run(invocation_context={'responses': ['first', 'second', 'third']})
    assert results['invocation_context']['documents'] == [Document(content='first', meta={'a': 'A'}), Document(content='second', meta={'a': 'A'}), Document(content='third', meta={'a': 'A'})]

@pytest.mark.unit
def test_strings_to_documents_wrong_number_of_meta():
    if False:
        for i in range(10):
            print('nop')
    shaper = Shaper(func='strings_to_documents', inputs={'strings': 'responses'}, params={'meta': [{'a': 'A'}]}, outputs=['documents'])
    with pytest.raises(ValueError, match='Not enough metadata dictionaries.'):
        shaper.run(invocation_context={'responses': ['first', 'second', 'third']})

@pytest.mark.unit
def test_strings_to_documents_many_meta_no_hashkeys():
    if False:
        return 10
    shaper = Shaper(func='strings_to_documents', inputs={'strings': 'responses'}, params={'meta': [{'a': i + 1} for i in range(3)]}, outputs=['documents'])
    (results, _) = shaper.run(invocation_context={'responses': ['first', 'second', 'third']})
    assert results['invocation_context']['documents'] == [Document(content='first', meta={'a': 1}), Document(content='second', meta={'a': 2}), Document(content='third', meta={'a': 3})]

@pytest.mark.unit
def test_strings_to_documents_single_meta_with_hashkeys():
    if False:
        for i in range(10):
            print('nop')
    shaper = Shaper(func='strings_to_documents', inputs={'strings': 'responses'}, params={'meta': {'a': 'A'}, 'id_hash_keys': ['content', 'meta']}, outputs=['documents'])
    (results, _) = shaper.run(invocation_context={'responses': ['first', 'second', 'third']})
    assert results['invocation_context']['documents'] == [Document(content='first', meta={'a': 'A'}, id_hash_keys=['content', 'meta']), Document(content='second', meta={'a': 'A'}, id_hash_keys=['content', 'meta']), Document(content='third', meta={'a': 'A'}, id_hash_keys=['content', 'meta'])]

@pytest.mark.unit
def test_strings_to_documents_no_meta_no_hashkeys_yaml(tmp_path):
    if False:
        print('Hello World!')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write("\n            version: ignore\n            components:\n            - name: shaper\n              type: Shaper\n              params:\n                func: strings_to_documents\n                params:\n                  strings: ['a', 'b', 'c']\n                outputs:\n                  - documents\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper\n                    inputs:\n                      - Query\n        ")
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run()
    assert result['invocation_context']['documents'] == [Document(content='a'), Document(content='b'), Document(content='c')]

@pytest.mark.unit
def test_strings_to_documents_meta_and_hashkeys_yaml(tmp_path):
    if False:
        while True:
            i = 10
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write("\n            version: ignore\n            components:\n            - name: shaper\n              type: Shaper\n              params:\n                func: strings_to_documents\n                params:\n                  strings: ['first', 'second', 'third']\n                  id_hash_keys: ['content', 'meta']\n                  meta:\n                    - a: 1\n                    - a: 2\n                    - a: 3\n                outputs:\n                  - documents\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper\n                    inputs:\n                      - Query\n        ")
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run()
    assert result['invocation_context']['documents'] == [Document(content='first', meta={'a': 1}, id_hash_keys=['content', 'meta']), Document(content='second', meta={'a': 2}, id_hash_keys=['content', 'meta']), Document(content='third', meta={'a': 3}, id_hash_keys=['content', 'meta'])]

@pytest.mark.unit
def test_documents_to_strings():
    if False:
        print('Hello World!')
    shaper = Shaper(func='documents_to_strings', inputs={'documents': 'documents'}, outputs=['strings'])
    (results, _) = shaper.run(documents=[Document(content='first'), Document(content='second'), Document(content='third')])
    assert results['invocation_context']['strings'] == ['first', 'second', 'third']

@pytest.mark.unit
def test_documents_to_strings_with_pattern_and_str_replace():
    if False:
        i = 10
        return i + 15
    shaper = Shaper(func='documents_to_strings', inputs={'documents': 'documents'}, outputs=['strings'], params={'pattern': '[$idx] $content', 'str_replace': {'r': 'R'}})
    (results, _) = shaper.run(documents=[Document(content='first'), Document(content='second'), Document(content='third')])
    assert results['invocation_context']['strings'] == ['[1] fiRst', '[2] second', '[3] thiRd']

@pytest.mark.unit
def test_documents_to_strings_yaml(tmp_path):
    if False:
        while True:
            i = 10
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: shaper\n              type: Shaper\n              params:\n                func: documents_to_strings\n                inputs:\n                  documents: documents\n                outputs:\n                  - strings\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper\n                    inputs:\n                      - Query\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(documents=[Document(content='a'), Document(content='b'), Document(content='c')])
    assert result['invocation_context']['strings'] == ['a', 'b', 'c']

@pytest.mark.unit
def test_documents_to_strings_with_pattern_and_str_replace_yaml(tmp_path):
    if False:
        while True:
            i = 10
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write("\n            version: ignore\n            components:\n            - name: shaper\n              type: Shaper\n              params:\n                func: documents_to_strings\n                inputs:\n                  documents: documents\n                outputs:\n                  - strings\n                params:\n                  pattern: '[$idx] $content'\n                  str_replace:\n                    r: R\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper\n                    inputs:\n                      - Query\n        ")
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(documents=[Document(content='first'), Document(content='second'), Document(content='third')])
    assert result['invocation_context']['strings'] == ['[1] fiRst', '[2] second', '[3] thiRd']

@pytest.mark.unit
def test_chain_shapers():
    if False:
        print('Hello World!')
    shaper_1 = Shaper(func='join_documents', inputs={'documents': 'documents'}, params={'delimiter': ' - '}, outputs=['documents'])
    shaper_2 = Shaper(func='value_to_list', inputs={'value': 'query', 'target_list': 'documents'}, outputs=['questions'])
    pipe = Pipeline()
    pipe.add_node(shaper_1, name='shaper_1', inputs=['Query'])
    pipe.add_node(shaper_2, name='shaper_2', inputs=['shaper_1'])
    results = pipe.run(query='test query', documents=[Document(content='first'), Document(content='second'), Document(content='third')])
    assert results['invocation_context']['documents'] == [Document(content='first - second - third')]
    assert results['invocation_context']['questions'] == ['test query']

@pytest.mark.unit
def test_chain_shapers_yaml(tmp_path):
    if False:
        i = 10
        return i + 15
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write("\n            version: ignore\n            components:\n\n            - name: shaper_1\n              type: Shaper\n              params:\n                func: join_documents\n                inputs:\n                  documents: documents\n                params:\n                  delimiter: ' - '\n                outputs:\n                  - documents\n\n            - name: shaper_2\n              type: Shaper\n              params:\n                func: value_to_list\n                inputs:\n                  value: query\n                  target_list: documents\n                outputs:\n                  - questions\n\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper_1\n                    inputs:\n                      - Query\n                  - name: shaper_2\n                    inputs:\n                      - shaper_1\n        ")
    pipe = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    results = pipe.run(query='test query', documents=[Document(content='first'), Document(content='second'), Document(content='third')])
    assert results['invocation_context']['documents'] == [Document(content='first - second - third')]
    assert results['invocation_context']['questions'] == ['test query']

@pytest.mark.unit
def test_chain_shapers_yaml_2(tmp_path):
    if False:
        print('Hello World!')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write("\n            version: ignore\n            components:\n\n            - name: shaper_1\n              type: Shaper\n              params:\n                func: strings_to_documents\n                params:\n                  strings:\n                    - first\n                    - second\n                    - third\n                outputs:\n                  - string_documents\n\n            - name: shaper_2\n              type: Shaper\n              params:\n                func: value_to_list\n                inputs:\n                  target_list: string_documents\n                params:\n                  value: hello\n                outputs:\n                  - greetings\n\n            - name: shaper_3\n              type: Shaper\n              params:\n                func: join_strings\n                inputs:\n                  strings: greetings\n                params:\n                  delimiter: '. '\n                outputs:\n                  - many_greetings\n\n            - name: expander\n              type: Shaper\n              params:\n                func: value_to_list\n                inputs:\n                  value: many_greetings\n                params:\n                  target_list: [1]\n                outputs:\n                  - many_greetings\n\n            - name: shaper_4\n              type: Shaper\n              params:\n                func: strings_to_documents\n                inputs:\n                  strings: many_greetings\n                outputs:\n                  - documents_with_greetings\n\n            pipelines:\n              - name: query\n                nodes:\n                  - name: shaper_1\n                    inputs:\n                      - Query\n                  - name: shaper_2\n                    inputs:\n                      - shaper_1\n                  - name: shaper_3\n                    inputs:\n                      - shaper_2\n                  - name: expander\n                    inputs:\n                      - shaper_3\n                  - name: shaper_4\n                    inputs:\n                      - expander\n        ")
    pipe = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    results = pipe.run()
    assert results['invocation_context']['documents_with_greetings'] == [Document(content='hello. hello. hello')]

@pytest.mark.integration
def test_with_prompt_node(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n              - name: prompt_model\n                type: PromptModel\n\n              - name: prompt_node\n                type: PromptNode\n                params:\n                  output_variable: answers\n                  model_name_or_path: prompt_model\n                  default_prompt_template: question-answering-per-document\n\n            pipelines:\n              - name: query\n                nodes:\n                  - name: prompt_node\n                    inputs:\n                      - Query\n            ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(query="What's Berlin like?", documents=[Document('Berlin is an amazing city.'), Document('Berlin is a cool city in Germany.')])
    assert len(result['answers']) == 2
    raw_answers = [answer.answer for answer in result['answers']]
    assert any((word for word in ['berlin', 'germany', 'cool', 'city', 'amazing'] if word in raw_answers))

@pytest.mark.integration
def test_with_multiple_prompt_nodes(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n              - name: prompt_model\n                type: PromptModel\n\n              - name: renamer\n                type: Shaper\n                params:\n                  func: rename\n                  inputs:\n                    value: new-questions\n                  outputs:\n                    - query\n\n              - name: prompt_node\n                type: PromptNode\n                params:\n                  model_name_or_path: prompt_model\n                  default_prompt_template: question-answering-per-document\n\n              - name: prompt_node_second\n                type: PromptNode\n                params:\n                  model_name_or_path: prompt_model\n                  default_prompt_template: question-generation\n                  output_variable: new-questions\n\n              - name: prompt_node_third\n                type: PromptNode\n                params:\n                  model_name_or_path: google/flan-t5-small\n                  default_prompt_template: question-answering-per-document\n\n            pipelines:\n              - name: query\n                nodes:\n                  - name: prompt_node\n                    inputs:\n                      - Query\n                  - name: prompt_node_second\n                    inputs:\n                      - prompt_node\n                  - name: renamer\n                    inputs:\n                      - prompt_node_second\n                  - name: prompt_node_third\n                    inputs:\n                      - renamer\n            ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(query="What's Berlin like?", documents=[Document('Berlin is an amazing city.'), Document('Berlin is a cool city in Germany.')])
    results = result['answers']
    assert len(results) == 2
    assert any((True for r in results if 'Berlin' in r.answer))

@pytest.mark.unit
def test_join_query_and_documents_yaml(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n\n            components:\n            - name: expander\n              type: Shaper\n              params:\n                func: value_to_list\n                inputs:\n                  value: query\n                params:\n                  target_list: [1]\n                outputs:\n                  - query\n\n            - name: joiner\n              type: Shaper\n              params:\n                func: join_lists\n                inputs:\n                  lists:\n                   - documents\n                   - query\n                outputs:\n                  - query\n\n            pipelines:\n              - name: query\n                nodes:\n                  - name: expander\n                    inputs:\n                      - Query\n                  - name: joiner\n                    inputs:\n                      - expander\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(query='What is going on here?', documents=['first', 'second', 'third'])
    assert result['query'] == ['first', 'second', 'third', 'What is going on here?']

@pytest.mark.unit
def test_join_query_and_documents_into_single_string_yaml(tmp_path):
    if False:
        return 10
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: expander\n              type: Shaper\n              params:\n                func: value_to_list\n                inputs:\n                  value: query\n                params:\n                  target_list: [1]\n                outputs:\n                  - query\n\n            - name: joiner\n              type: Shaper\n              params:\n                func: join_lists\n                inputs:\n                  lists:\n                   - documents\n                   - query\n                outputs:\n                  - query\n\n            - name: concatenator\n              type: Shaper\n              params:\n                func: join_strings\n                inputs:\n                  strings: query\n                outputs:\n                  - query\n\n            pipelines:\n              - name: query\n                nodes:\n                  - name: expander\n                    inputs:\n                      - Query\n                  - name: joiner\n                    inputs:\n                      - expander\n                  - name: concatenator\n                    inputs:\n                      - joiner\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(query='What is going on here?', documents=['first', 'second', 'third'])
    assert result['query'] == 'first second third What is going on here?'

@pytest.mark.unit
def test_join_query_and_documents_convert_into_documents_yaml(tmp_path):
    if False:
        i = 10
        return i + 15
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: expander\n              type: Shaper\n              params:\n                func: value_to_list\n                inputs:\n                  value: query\n                params:\n                  target_list: [1]\n                outputs:\n                  - query\n\n            - name: joiner\n              type: Shaper\n              params:\n                func: join_lists\n                inputs:\n                  lists:\n                   - documents\n                   - query\n                outputs:\n                  - query_and_docs\n\n            - name: converter\n              type: Shaper\n              params:\n                func: strings_to_documents\n                inputs:\n                  strings: query_and_docs\n                outputs:\n                  - query_and_docs\n\n            pipelines:\n              - name: query\n                nodes:\n                  - name: expander\n                    inputs:\n                      - Query\n                  - name: joiner\n                    inputs:\n                      - expander\n                  - name: converter\n                    inputs:\n                      - joiner\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    result = pipeline.run(query='What is going on here?', documents=['first', 'second', 'third'])
    assert result['invocation_context']['query_and_docs']
    assert len(result['invocation_context']['query_and_docs']) == 4
    assert isinstance(result['invocation_context']['query_and_docs'][0], Document)

@pytest.mark.unit
def test_shaper_publishes_unknown_arg_does_not_break_pipeline():
    if False:
        return 10
    documents = [Document(content='test query')]
    shaper = Shaper(func='rename', inputs={'value': 'query'}, outputs=['unknown_by_retriever'], publish_outputs=True)
    document_store = InMemoryDocumentStore(use_bm25=True)
    document_store.write_documents(documents)
    retriever = BM25Retriever(document_store=document_store)
    pipeline = Pipeline()
    pipeline.add_node(component=shaper, name='shaper', inputs=['Query'])
    pipeline.add_node(component=retriever, name='retriever', inputs=['shaper'])
    result = pipeline.run(query='test query')
    assert result['invocation_context']['unknown_by_retriever'] == 'test query'
    assert result['unknown_by_retriever'] == 'test query'
    assert len(result['documents']) == 1