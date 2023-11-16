from typing import Dict, List
from haystack.preview import component, Document
from haystack.preview.utils.filters import document_matches_filter

@component
class MetadataRouter:
    """
    A component that routes documents to different connections based on the content of their fields.
    """

    def __init__(self, rules: Dict[str, Dict]):
        if False:
            return 10
        '\n        Initialize the MetadataRouter.\n\n        :param rules: A dictionary of rules that specify which edge to route a document to based on its metadata.\n                      The keys of the dictionary are the names of the output connections, and the values are dictionaries that\n                      follow the format of filtering expressions in Haystack. For example:\n                      ```python\n                      {\n                            "edge_1": {"created_at": {"$gte": "2023-01-01", "$lt": "2023-04-01"}},\n                            "edge_2": {"created_at": {"$gte": "2023-04-01", "$lt": "2023-07-01"}},\n                            "edge_3": {"created_at": {"$gte": "2023-07-01", "$lt": "2023-10-01"}},\n                            "edge_4": {"created_at": {"$gte": "2023-10-01", "$lt": "2024-01-01"}},\n                      }\n                      ```\n        '
        self.rules = rules
        component.set_output_types(self, unmatched=List[Document], **{edge: List[Document] for edge in rules})

    def run(self, documents: List[Document]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Run the MetadataRouter. This method routes the documents to different edges based on their fields content and\n        the rules specified during initialization. If a document does not match any of the rules, it is routed to\n        a connection named "unmatched".\n\n        :param documents: A list of documents to route to different edges.\n        '
        unmatched_documents = []
        output: Dict[str, List[Document]] = {edge: [] for edge in self.rules}
        for document in documents:
            cur_document_matched = False
            for (edge, rule) in self.rules.items():
                if document_matches_filter(rule, document):
                    output[edge].append(document)
                    cur_document_matched = True
            if not cur_document_matched:
                unmatched_documents.append(document)
        output['unmatched'] = unmatched_documents
        return output