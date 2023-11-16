from abc import abstractmethod
import logging
from numpy import mat
import pytest
import json
import inspect
import networkx as nx
from enum import Enum
from pydantic.dataclasses import dataclass
from typing import Any, Dict, List, Optional
import haystack
from haystack import Pipeline
from haystack.nodes import _json_schema
from haystack.nodes import FileTypeClassifier
from haystack.errors import HaystackError, PipelineConfigError, PipelineSchemaError, DocumentStoreError
from haystack.nodes.base import BaseComponent
from ..conftest import MockNode, MockDocumentStore, MockReader, MockRetriever
from .. import conftest

@pytest.fixture(autouse=True)
def mock_json_schema(request, monkeypatch, tmp_path):
    if False:
        print('Hello World!')
    '\n    JSON schema with the main version and only mocked nodes.\n    '
    if 'integration' in request.keywords:
        return
    monkeypatch.setattr(haystack.nodes._json_schema, 'find_subclasses_in_modules', lambda *a, **k: [(conftest, MockDocumentStore), (conftest, MockReader), (conftest, MockRetriever)])
    monkeypatch.setattr(haystack.nodes._json_schema, 'JSON_SCHEMAS_PATH', tmp_path)
    filename = 'haystack-pipeline-main.schema.json'
    test_schema = _json_schema.get_json_schema(filename=filename, version='ignore')
    with open(tmp_path / filename, 'w') as schema_file:
        json.dump(test_schema, schema_file, indent=4)

@pytest.mark.integration
@pytest.mark.elasticsearch
def test_load_and_save_from_yaml(tmp_path, samples_path):
    if False:
        return 10
    config_path = samples_path / 'pipeline' / 'test.haystack-pipeline.yml'
    indexing_pipeline = Pipeline.load_from_yaml(path=config_path, pipeline_name='indexing_pipeline')
    indexing_pipeline.get_document_store().delete_documents()
    assert indexing_pipeline.get_document_store().get_document_count() == 0
    indexing_pipeline.run(file_paths=samples_path / 'pdf' / 'sample_pdf_1.pdf')
    assert indexing_pipeline.get_document_store().get_document_count() > 0
    new_indexing_config = tmp_path / 'test_indexing.yaml'
    indexing_pipeline.save_to_yaml(new_indexing_config)
    new_indexing_pipeline = Pipeline.load_from_yaml(path=new_indexing_config)
    assert nx.is_isomorphic(new_indexing_pipeline.graph, indexing_pipeline.graph)
    modified_indexing_pipeline = Pipeline.load_from_yaml(path=new_indexing_config)
    modified_indexing_pipeline.add_node(FileTypeClassifier(), name='file_classifier', inputs=['File'])
    assert not nx.is_isomorphic(new_indexing_pipeline.graph, modified_indexing_pipeline.graph)
    query_pipeline = Pipeline.load_from_yaml(path=config_path, pipeline_name='query_pipeline')
    prediction = query_pipeline.run(query='Who made the PDF specification?', params={'ESRetriever': {'top_k': 10}, 'Reader': {'top_k': 3}})
    assert prediction['query'] == 'Who made the PDF specification?'
    assert prediction['answers'][0].answer == 'Adobe Systems'
    assert '_debug' not in prediction.keys()
    new_query_config = tmp_path / 'test_query.yaml'
    query_pipeline.save_to_yaml(new_query_config)
    new_query_pipeline = Pipeline.load_from_yaml(path=new_query_config)
    assert nx.is_isomorphic(new_query_pipeline.graph, query_pipeline.graph)
    assert not nx.is_isomorphic(new_query_pipeline.graph, new_indexing_pipeline.graph)

@pytest.mark.unit
def test_load_yaml(tmp_path):
    if False:
        print('Hello World!')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: retriever\n              type: MockRetriever\n            - name: reader\n              type: MockReader\n            pipelines:\n            - name: query\n              nodes:\n              - name: retriever\n                inputs:\n                - Query\n              - name: reader\n                inputs:\n                - retriever\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    assert len(pipeline.graph.nodes) == 3
    assert isinstance(pipeline.get_node('retriever'), MockRetriever)
    assert isinstance(pipeline.get_node('reader'), MockReader)

def test_load_yaml_elasticsearch_not_responding(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: ESRetriever\n              type: BM25Retriever\n              params:\n                document_store: DocumentStore\n            - name: DocumentStore\n              type: ElasticsearchDocumentStore\n              params:\n                port: 1234\n            - name: PDFConverter\n              type: PDFToTextConverter\n            - name: Preprocessor\n              type: PreProcessor\n            pipelines:\n            - name: query_pipeline\n              nodes:\n              - name: ESRetriever\n                inputs: [Query]\n            - name: indexing_pipeline\n              nodes:\n              - name: PDFConverter\n                inputs: [File]\n              - name: Preprocessor\n                inputs: [PDFConverter]\n              - name: ESRetriever\n                inputs: [Preprocessor]\n              - name: DocumentStore\n                inputs: [ESRetriever]\n        ')
    with pytest.raises(DocumentStoreError):
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml', pipeline_name='indexing_pipeline')

@pytest.mark.unit
def test_load_yaml_non_existing_file(samples_path):
    if False:
        i = 10
        return i + 15
    with pytest.raises(FileNotFoundError):
        Pipeline.load_from_yaml(path=samples_path / 'pipeline' / 'I_dont_exist.yml')

@pytest.mark.unit
def test_load_yaml_invalid_yaml(tmp_path):
    if False:
        while True:
            i = 10
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('this is not valid YAML!')
    with pytest.raises(PipelineConfigError):
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')

@pytest.mark.unit
def test_load_yaml_missing_version(tmp_path):
    if False:
        return 10
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            components:\n            - name: docstore\n              type: MockDocumentStore\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: docstore\n                inputs:\n                - Query\n        ')
    with pytest.raises(PipelineConfigError, match='Validation failed') as e:
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
        assert 'version' in str(e)

@pytest.mark.unit
def test_load_yaml_non_existing_version(tmp_path, caplog):
    if False:
        return 10
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: random\n            components:\n            - name: docstore\n              type: MockDocumentStore\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: docstore\n                inputs:\n                - Query\n        ')
    with caplog.at_level(logging.WARNING):
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
        assert "version 'random'" in caplog.text
        assert f'Haystack {haystack.__version__}' in caplog.text

@pytest.mark.unit
def test_load_yaml_non_existing_version_strict(tmp_path):
    if False:
        return 10
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: random\n            components:\n            - name: docstore\n              type: MockDocumentStore\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: docstore\n                inputs:\n                - Query\n        ')
    with pytest.raises(PipelineConfigError, match='Cannot load pipeline configuration of version random'):
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml', strict_version_check=True)

@pytest.mark.unit
def test_load_yaml_incompatible_version(tmp_path, caplog):
    if False:
        i = 10
        return i + 15
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: 1.1.0\n            components:\n            - name: docstore\n              type: MockDocumentStore\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: docstore\n                inputs:\n                - Query\n        ')
    with caplog.at_level(logging.WARNING):
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
        assert "version '1.1.0'" in caplog.text
        assert f'Haystack {haystack.__version__}' in caplog.text

@pytest.mark.unit
def test_load_yaml_incompatible_version_strict(tmp_path):
    if False:
        i = 10
        return i + 15
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: 1.1.0\n            components:\n            - name: docstore\n              type: MockDocumentStore\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: docstore\n                inputs:\n                - Query\n        ')
    with pytest.raises(PipelineConfigError, match='Cannot load pipeline configuration of version 1.1.0'):
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml', strict_version_check=True)

@pytest.mark.unit
def test_load_yaml_no_components(tmp_path):
    if False:
        i = 10
        return i + 15
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            pipelines:\n            - name: my_pipeline\n              nodes:\n        ')
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
        assert 'components' in str(e)

@pytest.mark.unit
def test_load_yaml_wrong_component(tmp_path):
    if False:
        i = 10
        return i + 15
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: docstore\n              type: ImaginaryDocumentStore\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: docstore\n                inputs:\n                - Query\n        ')
    with pytest.raises(HaystackError) as e:
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
        assert 'ImaginaryDocumentStore' in str(e)

@pytest.mark.unit
def test_load_yaml_custom_component(tmp_path):
    if False:
        return 10

    class CustomNode(MockNode):

        def __init__(self, param: int):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.param = param
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: custom_node\n              type: CustomNode\n              params:\n                param: 1\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: custom_node\n                inputs:\n                - Query\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    assert pipeline.get_node('custom_node').param == 1

@pytest.mark.unit
def test_load_yaml_custom_component_with_null_values(tmp_path):
    if False:
        print('Hello World!')

    class CustomNode(MockNode):

        def __init__(self, param: Optional[str], lst_param: Optional[List[Any]], dict_param: Optional[Dict[str, Any]]):
            if False:
                return 10
            super().__init__()
            self.param = param
            self.lst_param = lst_param
            self.dict_param = dict_param
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: custom_node\n              type: CustomNode\n              params:\n                param: null\n                lst_param: null\n                dict_param: null\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: custom_node\n                inputs:\n                - Query\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    assert len(pipeline.graph.nodes) == 2
    assert pipeline.get_node('custom_node').param is None
    assert pipeline.get_node('custom_node').lst_param is None
    assert pipeline.get_node('custom_node').dict_param is None

@pytest.mark.unit
def test_load_yaml_custom_component_with_no_init(tmp_path):
    if False:
        print('Hello World!')

    class CustomNode(MockNode):
        pass
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: custom_node\n              type: CustomNode\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: custom_node\n                inputs:\n                - Query\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    assert isinstance(pipeline.get_node('custom_node'), CustomNode)

@pytest.mark.unit
def test_load_yaml_custom_component_neednt_call_super(tmp_path):
    if False:
        while True:
            i = 10
    'This is a side-effect. Here for behavior documentation only'

    class CustomNode(BaseComponent):
        outgoing_edges = 1

        def __init__(self, param: int):
            if False:
                i = 10
                return i + 15
            self.param = param

        def run(self, *a, **k):
            if False:
                return 10
            pass

        def run_batch(self, *a, **k):
            if False:
                print('Hello World!')
            pass
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: custom_node\n              type: CustomNode\n              params:\n                param: 1\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: custom_node\n                inputs:\n                - Query\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    assert isinstance(pipeline.get_node('custom_node'), CustomNode)
    assert pipeline.get_node('custom_node').param == 1

@pytest.mark.unit
def test_load_yaml_custom_component_cant_be_abstract(tmp_path):
    if False:
        print('Hello World!')

    class CustomNode(MockNode):

        @abstractmethod
        def abstract_method(self):
            if False:
                print('Hello World!')
            pass
    assert inspect.isabstract(CustomNode)
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: custom_node\n              type: CustomNode\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: custom_node\n                inputs:\n                - Query\n        ')
    with pytest.raises(PipelineSchemaError, match='abstract'):
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')

@pytest.mark.unit
def test_load_yaml_custom_component_name_can_include_base(tmp_path):
    if False:
        for i in range(10):
            print('nop')

    class BaseCustomNode(MockNode):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            super().__init__()
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: custom_node\n              type: BaseCustomNode\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: custom_node\n                inputs:\n                - Query\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    assert isinstance(pipeline.get_node('custom_node'), BaseCustomNode)

@pytest.mark.unit
def test_load_yaml_custom_component_must_subclass_basecomponent(tmp_path):
    if False:
        print('Hello World!')

    class SomeCustomNode:

        def run(self, *a, **k):
            if False:
                return 10
            pass
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: custom_node\n              type: SomeCustomNode\n              params:\n                param: 1\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: custom_node\n                inputs:\n                - Query\n        ')
    with pytest.raises(PipelineSchemaError, match="'SomeCustomNode' not found"):
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')

@pytest.mark.unit
def test_load_yaml_custom_component_referencing_other_node_in_init(tmp_path):
    if False:
        i = 10
        return i + 15

    class OtherNode(MockNode):

        def __init__(self, another_param: str):
            if False:
                print('Hello World!')
            super().__init__()
            self.param = another_param

    class CustomNode(MockNode):

        def __init__(self, other_node: OtherNode):
            if False:
                return 10
            super().__init__()
            self.other_node = other_node
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: other_node\n              type: OtherNode\n              params:\n                another_param: value\n            - name: custom_node\n              type: CustomNode\n              params:\n                other_node: other_node\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: custom_node\n                inputs:\n                - Query\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    assert isinstance(pipeline.get_node('custom_node'), CustomNode)
    assert isinstance(pipeline.get_node('custom_node').other_node, OtherNode)
    assert pipeline.get_node('custom_node').name == 'custom_node'
    assert pipeline.get_node('custom_node').other_node.name == 'other_node'

@pytest.mark.unit
def test_load_yaml_custom_component_with_helper_class_in_init(tmp_path):
    if False:
        print('Hello World!')
    '\n    This test can work from the perspective of YAML schema validation:\n    HelperClass is picked up correctly and everything gets loaded.\n\n    However, for now we decide to disable this feature.\n    See haystack/_json_schema.py for details.\n    '

    @dataclass
    class HelperClass:

        def __init__(self, another_param: str):
            if False:
                return 10
            self.param = another_param

    class CustomNode(MockNode):

        def __init__(self, some_exotic_parameter: HelperClass=HelperClass(1)):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.some_exotic_parameter = some_exotic_parameter
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: custom_node\n              type: CustomNode\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: custom_node\n                inputs:\n                - Query\n        ')
    with pytest.raises(PipelineSchemaError, match='takes object instances as parameters in its __init__ function'):
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')

@pytest.mark.unit
def test_load_yaml_custom_component_with_helper_class_in_yaml(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    This test can work from the perspective of YAML schema validation:\n    HelperClass is picked up correctly and everything gets loaded.\n\n    However, for now we decide to disable this feature.\n    See haystack/_json_schema.py for details.\n    '

    class HelperClass:

        def __init__(self, another_param: str):
            if False:
                while True:
                    i = 10
            self.param = another_param

    class CustomNode(MockNode):

        def __init__(self, some_exotic_parameter: HelperClass):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.some_exotic_parameter = some_exotic_parameter
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: custom_node\n              type: CustomNode\n              params:\n                some_exotic_parameter: HelperClass("hello")\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: custom_node\n                inputs:\n                - Query\n        ')
    pipe = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    assert pipe.get_node('custom_node').some_exotic_parameter == 'HelperClass("hello")'

@pytest.mark.unit
def test_load_yaml_custom_component_with_enum_in_init(tmp_path):
    if False:
        return 10
    '\n    This test can work from the perspective of YAML schema validation:\n    Flags is picked up correctly and everything gets loaded.\n\n    However, for now we decide to disable this feature.\n    See haystack/_json_schema.py for details.\n    '

    class Flags(Enum):
        FIRST_VALUE = 1
        SECOND_VALUE = 2

    class CustomNode(MockNode):

        def __init__(self, some_exotic_parameter: Flags=None):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.some_exotic_parameter = some_exotic_parameter
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: custom_node\n              type: CustomNode\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: custom_node\n                inputs:\n                - Query\n        ')
    with pytest.raises(PipelineSchemaError, match='takes object instances as parameters in its __init__ function'):
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')

@pytest.mark.unit
def test_load_yaml_custom_component_with_enum_in_yaml(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    This test can work from the perspective of YAML schema validation:\n    Flags is picked up correctly and everything gets loaded.\n\n    However, for now we decide to disable this feature.\n    See haystack/_json_schema.py for details.\n    '

    class Flags(Enum):
        FIRST_VALUE = 1
        SECOND_VALUE = 2

    class CustomNode(MockNode):

        def __init__(self, some_exotic_parameter: Flags):
            if False:
                while True:
                    i = 10
            super().__init__()
            self.some_exotic_parameter = some_exotic_parameter
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: custom_node\n              type: CustomNode\n              params:\n                some_exotic_parameter: Flags.SECOND_VALUE\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: custom_node\n                inputs:\n                - Query\n        ')
    with pytest.raises(PipelineSchemaError, match='takes object instances as parameters in its __init__ function'):
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')

@pytest.mark.unit
def test_load_yaml_custom_component_with_external_constant(tmp_path):
    if False:
        i = 10
        return i + 15
    '\n    This is a potential pitfall. The code should work as described here.\n    '

    class AnotherClass:
        CLASS_CONSTANT = 'str'

    class CustomNode(MockNode):

        def __init__(self, some_exotic_parameter: str):
            if False:
                return 10
            super().__init__()
            self.some_exotic_parameter = some_exotic_parameter
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: custom_node\n              type: CustomNode\n              params:\n                some_exotic_parameter: AnotherClass.CLASS_CONSTANT  # Will *NOT* be resolved\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: custom_node\n                inputs:\n                - Query\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    node = pipeline.get_node('custom_node')
    assert node.some_exotic_parameter == 'AnotherClass.CLASS_CONSTANT'

@pytest.mark.unit
def test_load_yaml_custom_component_with_superclass(tmp_path):
    if False:
        while True:
            i = 10

    class BaseCustomNode(MockNode):

        def __init__(self):
            if False:
                return 10
            super().__init__()

    class CustomNode(BaseCustomNode):

        def __init__(self, some_exotic_parameter: str):
            if False:
                return 10
            super().__init__()
            self.some_exotic_parameter = some_exotic_parameter
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: custom_node\n              type: CustomNode\n              params:\n                some_exotic_parameter: value\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: custom_node\n                inputs:\n                - Query\n        ')
    Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')

@pytest.mark.unit
def test_load_yaml_custom_component_with_variadic_args(tmp_path):
    if False:
        for i in range(10):
            print('nop')

    class BaseCustomNode(MockNode):

        def __init__(self, base_parameter: int):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.base_parameter = base_parameter

    class CustomNode(BaseCustomNode):

        def __init__(self, some_parameter: str, *args):
            if False:
                while True:
                    i = 10
            super().__init__(*args)
            self.some_parameter = some_parameter
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: custom_node\n              type: CustomNode\n              params:\n                base_parameter: 1\n                some_parameter: value\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: custom_node\n                inputs:\n                - Query\n        ')
    with pytest.raises(PipelineSchemaError, match='variadic'):
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')

@pytest.mark.unit
def test_load_yaml_custom_component_with_variadic_kwargs(tmp_path):
    if False:
        while True:
            i = 10

    class BaseCustomNode(MockNode):

        def __init__(self, base_parameter: int):
            if False:
                return 10
            super().__init__()
            self.base_parameter = base_parameter

    class CustomNode(BaseCustomNode):

        def __init__(self, some_parameter: str, **kwargs):
            if False:
                while True:
                    i = 10
            super().__init__(**kwargs)
            self.some_parameter = some_parameter
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: custom_node\n              type: CustomNode\n              params:\n                base_parameter: 1\n                some_parameter: value\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: custom_node\n                inputs:\n                - Query\n        ')
    with pytest.raises(PipelineSchemaError, match='variadic'):
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')

@pytest.mark.unit
def test_load_yaml_no_pipelines(tmp_path):
    if False:
        return 10
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: docstore\n              type: MockDocumentStore\n            pipelines:\n        ')
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
        assert 'pipeline' in str(e)

@pytest.mark.unit
def test_load_yaml_invalid_pipeline_name(tmp_path):
    if False:
        print('Hello World!')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: docstore\n              type: MockDocumentStore\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: docstore\n                inputs:\n                - Query\n        ')
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml', pipeline_name='invalid')
        assert 'invalid' in str(e) and 'pipeline' in str(e)

@pytest.mark.unit
def test_load_yaml_pipeline_with_wrong_nodes(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: docstore\n              type: MockDocumentStore\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: not_existing_node\n                inputs:\n                - Query\n        ')
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
        assert 'not_existing_node' in str(e)

@pytest.mark.unit
def test_load_yaml_pipeline_not_acyclic_graph(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: retriever\n              type: MockRetriever\n            - name: reader\n              type: MockRetriever\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: retriever\n                inputs:\n                - reader\n              - name: reader\n                inputs:\n                - retriever\n        ')
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
        assert 'reader' in str(e) or 'retriever' in str(e)
        assert 'loop' in str(e)

@pytest.mark.unit
def test_load_yaml_wrong_root(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: retriever\n              type: MockRetriever\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: retriever\n                inputs:\n                - Nothing\n        ')
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
        assert 'Nothing' in str(e)
        assert 'root' in str(e).lower()

@pytest.mark.unit
def test_load_yaml_two_roots_invalid(tmp_path):
    if False:
        return 10
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: retriever\n              type: MockRetriever\n            - name: retriever_2\n              type: MockRetriever\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: retriever\n                inputs:\n                - Query\n              - name: retriever_2\n                inputs:\n                - File\n        ')
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    assert 'File' in str(e) or 'Query' in str(e)

@pytest.mark.unit
def test_load_yaml_two_roots_valid(tmp_path):
    if False:
        print('Hello World!')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: retriever\n              type: MockRetriever\n            - name: retriever_2\n              type: MockRetriever\n            pipelines:\n            - name: my_pipeline\n              nodes:\n              - name: retriever\n                inputs:\n                - Query\n              - name: retriever_2\n                inputs:\n                - Query\n        ')
    Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')

@pytest.mark.unit
def test_load_yaml_two_roots_in_separate_pipelines(tmp_path):
    if False:
        i = 10
        return i + 15
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: node_1\n              type: MockNode\n            - name: node_2\n              type: MockNode\n            pipelines:\n            - name: pipeline_1\n              nodes:\n              - name: node_1\n                inputs:\n                - Query\n              - name: node_2\n                inputs:\n                - Query\n            - name: pipeline_2\n              nodes:\n              - name: node_1\n                inputs:\n                - File\n              - name: node_2\n                inputs:\n                - File\n        ')
    Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml', pipeline_name='pipeline_1')
    Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml', pipeline_name='pipeline_2')

@pytest.mark.unit
def test_load_yaml_disconnected_component(tmp_path):
    if False:
        print('Hello World!')
    with open(tmp_path / 'tmp_config.yml', 'w') as tmp_file:
        tmp_file.write('\n            version: ignore\n            components:\n            - name: docstore\n              type: MockDocumentStore\n            - name: retriever\n              type: MockRetriever\n            pipelines:\n            - name: query\n              nodes:\n              - name: docstore\n                inputs:\n                - Query\n        ')
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    assert len(pipeline.graph.nodes) == 2
    assert isinstance(pipeline.get_document_store(), MockDocumentStore)
    assert not pipeline.get_node('retriever')

@pytest.mark.unit
def test_load_yaml_unusual_chars_in_values(tmp_path):
    if False:
        while True:
            i = 10

    class DummyNode(BaseComponent):
        outgoing_edges = 1

        def __init__(self, space_param, non_alphanumeric_param):
            if False:
                return 10
            super().__init__()
            self.space_param = space_param
            self.non_alphanumeric_param = non_alphanumeric_param

        def run(self):
            if False:
                print('Hello World!')
            raise NotImplementedError

        def run_batch(self):
            if False:
                return 10
            raise NotImplementedError
    with open(tmp_path / 'tmp_config.yml', 'w', encoding='utf-8') as tmp_file:
        tmp_file.write("\n            version: '1.9.0'\n\n            components:\n                - name: DummyNode\n                  type: DummyNode\n                  params:\n                    space_param: with space\n                    non_alphanumeric_param: \\[ümlaut\\]\n\n            pipelines:\n                - name: indexing\n                  nodes:\n                    - name: DummyNode\n                      inputs: [File]\n        ")
    pipeline = Pipeline.load_from_yaml(path=tmp_path / 'tmp_config.yml')
    assert pipeline.components['DummyNode'].space_param == 'with space'
    assert pipeline.components['DummyNode'].non_alphanumeric_param == '\\[ümlaut\\]'

@pytest.mark.unit
def test_save_yaml(tmp_path):
    if False:
        while True:
            i = 10
    pipeline = Pipeline()
    pipeline.add_node(MockRetriever(), name='retriever', inputs=['Query'])
    pipeline.save_to_yaml(tmp_path / 'saved_pipeline.yml')
    with open(tmp_path / 'saved_pipeline.yml', 'r') as saved_yaml:
        content = saved_yaml.read()
        assert content.count('retriever') == 2
        assert 'MockRetriever' in content
        assert 'Query' in content
        assert f'version: {haystack.__version__}' in content

@pytest.mark.unit
def test_save_yaml_overwrite(tmp_path):
    if False:
        while True:
            i = 10
    pipeline = Pipeline()
    retriever = MockRetriever()
    pipeline.add_node(component=retriever, name='retriever', inputs=['Query'])
    with open(tmp_path / 'saved_pipeline.yml', 'w') as _:
        pass
    pipeline.save_to_yaml(tmp_path / 'saved_pipeline.yml')
    with open(tmp_path / 'saved_pipeline.yml', 'r') as saved_yaml:
        content = saved_yaml.read()
        assert content != ''

@pytest.mark.unit
@pytest.mark.parametrize('pipeline_file', ['ray.simple.haystack-pipeline.yml', 'ray.advanced.haystack-pipeline.yml'])
def test_load_yaml_ray_args_in_pipeline(samples_path, pipeline_file):
    if False:
        i = 10
        return i + 15
    with pytest.raises(PipelineConfigError):
        Pipeline.load_from_yaml(samples_path / 'pipeline' / pipeline_file, pipeline_name='ray_query_pipeline')