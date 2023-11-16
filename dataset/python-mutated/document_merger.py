import logging
from copy import deepcopy
from typing import Optional, List, Dict, Union, Any
from haystack.schema import Document
from haystack.nodes.base import BaseComponent
logger = logging.getLogger(__name__)

class DocumentMerger(BaseComponent):
    """
    A node to merge the texts of the documents.
    """
    outgoing_edges = 1

    def __init__(self, separator: str=' '):
        if False:
            while True:
                i = 10
        '\n        :param separator: The separator that appears between subsequent merged documents.\n        '
        super().__init__()
        self.separator = separator

    def merge(self, documents: List[Document], separator: Optional[str]=None) -> List[Document]:
        if False:
            i = 10
            return i + 15
        '\n        Produce a list made up of a single document, which contains all the texts of the documents provided.\n\n        :param separator: The separator that appears between subsequent merged documents.\n        :return: List of Documents\n        '
        if len(documents) == 0:
            raise ValueError('Document Merger needs at least one document to merge.')
        if not all((doc.content_type == 'text' for doc in documents)):
            raise ValueError('Some of the documents provided are non-textual. Document Merger only works on textual documents.')
        separator = separator if separator is not None else self.separator
        merged_content = separator.join([doc.content for doc in documents])
        common_meta = self._keep_common_keys([doc.meta for doc in documents])
        merged_document = Document(content=merged_content, meta=common_meta)
        return [merged_document]

    def run(self, documents: List[Document], separator: Optional[str]=None):
        if False:
            print('Hello World!')
        results: Dict = {'documents': []}
        if documents:
            results['documents'] = self.merge(documents=documents, separator=separator)
        return (results, 'output_1')

    def run_batch(self, documents: Union[List[Document], List[List[Document]]], separator: Optional[str]=None):
        if False:
            print('Hello World!')
        is_doclist_flat = isinstance(documents[0], Document)
        if is_doclist_flat:
            flat_result: List[Document] = self.merge(documents=[doc for doc in documents if isinstance(doc, Document)], separator=separator)
            return ({'documents': flat_result}, 'output_1')
        else:
            nested_result: List[List[Document]] = [self.merge(documents=docs_lst, separator=separator) for docs_lst in documents if isinstance(docs_lst, list)]
            return ({'documents': nested_result}, 'output_1')

    def _keep_common_keys(self, list_of_dicts: List[Dict[str, Any]]) -> dict:
        if False:
            return 10
        merge_dictionary = deepcopy(list_of_dicts[0])
        for (key, value) in list_of_dicts[0].items():
            if not all((key in dict.keys() for dict in list_of_dicts)):
                del merge_dictionary[key]
            elif isinstance(value, dict):
                list_of_subdicts = [dictionary[key] for dictionary in list_of_dicts]
                merge_dictionary[key] = self._keep_common_keys(list_of_subdicts)
            elif not all((value == dict[key] for dict in list_of_dicts)):
                del merge_dictionary[key]
        return merge_dictionary