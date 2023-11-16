import logging
import re
from typing import List, Dict, Any, Optional
from haystack.preview import component, GeneratedAnswer, Document
logger = logging.getLogger(__name__)

@component
class AnswerBuilder:
    """
    A component to parse the output of a Generator to `Answer` objects using regular expressions.
    """

    def __init__(self, pattern: Optional[str]=None, reference_pattern: Optional[str]=None):
        if False:
            while True:
                i = 10
        '\n        :param pattern: The regular expression pattern to use to extract the answer text from the generator output.\n                        If not specified, the whole string is used as the answer. The regular expression can have at\n                        most one capture group. If a capture group is present, the text matched by the capture group\n                        is used as the answer. If no capture group is present, the whole match is used as the answer.\n                        Examples:\n                            `[^\\n]+$` finds "this is an answer" in a string "this is an argument.\nthis is an answer".\n                            `Answer: (.*)` finds "this is an answer" in a string "this is an argument. Answer: this is an answer".\n                        Default: `None`.\n        :param reference_pattern: The regular expression pattern to use for parsing the document references.\n                                  We assume that references are specified as indices of the input documents and that\n                                  indices start at 1.\n                                  Example: `\\[(\\d+)\\]` finds "1" in a string "this is an answer[1]".\n                                  If not specified, no parsing is done, and all documents are referenced.\n                                  Default: `None`.\n        '
        if pattern:
            AnswerBuilder._check_num_groups_in_regex(pattern)
        self.pattern = pattern
        self.reference_pattern = reference_pattern

    @component.output_types(answers=List[GeneratedAnswer])
    def run(self, query: str, replies: List[str], metadata: Optional[List[Dict[str, Any]]]=None, documents: Optional[List[Document]]=None, pattern: Optional[str]=None, reference_pattern: Optional[str]=None):
        if False:
            print('Hello World!')
        '\n        Parse the output of a Generator to `Answer` objects using regular expressions.\n\n        :param query: The query used in the prompts for the Generator. A strings.\n        :param replies: The output of the Generator. A list of strings.\n        :param metadata: The metadata returned by the Generator. An optional list of dictionaries. If not specified,\n                            the generated answer will contain no metadata.\n        :param documents: The documents used as input to the Generator. A list of `Document` objects. If\n                          `documents` are specified, they are added to the `Answer` objects.\n                          If both `documents` and `reference_pattern` are specified, the documents referenced in the\n                          Generator output are extracted from the input documents and added to the `Answer` objects.\n                          Default: `None`.\n        :param pattern: The regular expression pattern to use to extract the answer text from the generator output.\n                        If not specified, the whole string is used as the answer. The regular expression can have at\n                        most one capture group. If a capture group is present, the text matched by the capture group\n                        is used as the answer. If no capture group is present, the whole match is used as the answer.\n                        Examples:\n                            `[^\\n]+$` finds "this is an answer" in a string "this is an argument.\nthis is an answer".\n                            `Answer: (.*)` finds "this is an answer" in a string "this is an argument. Answer: this is an answer".\n                        Default: `None`.\n        :param reference_pattern: The regular expression pattern to use for parsing the document references.\n                                  We assume that references are specified as indices of the input documents and that\n                                  indices start at 1.\n                                  Example: `\\[(\\d+)\\]` finds "1" in a string "this is an answer[1]".\n                                  If not specified, no parsing is done, and all documents are referenced.\n                                  Default: `None`.\n        '
        if not metadata:
            metadata = [{}] * len(replies)
        elif len(replies) != len(metadata):
            raise ValueError(f'Number of replies ({len(replies)}), and metadata ({len(metadata)}) must match.')
        if pattern:
            AnswerBuilder._check_num_groups_in_regex(pattern)
        pattern = pattern or self.pattern
        reference_pattern = reference_pattern or self.reference_pattern
        all_answers = []
        for (reply, meta) in zip(replies, metadata):
            referenced_docs = []
            if documents:
                reference_idxs = []
                if reference_pattern:
                    reference_idxs = AnswerBuilder._extract_reference_idxs(reply, reference_pattern)
                else:
                    reference_idxs = [doc_idx for (doc_idx, _) in enumerate(documents)]
                for idx in reference_idxs:
                    try:
                        referenced_docs.append(documents[idx])
                    except IndexError:
                        logger.warning("Document index '%s' referenced in Generator output is out of range. ", idx + 1)
            answer_string = AnswerBuilder._extract_answer_string(reply, pattern)
            answer = GeneratedAnswer(data=answer_string, query=query, documents=referenced_docs, metadata=meta)
            all_answers.append(answer)
        return {'answers': all_answers}

    @staticmethod
    def _extract_answer_string(reply: str, pattern: Optional[str]=None) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Extract the answer string from the generator output using the specified pattern.\n        If no pattern is specified, the whole string is used as the answer.\n\n        :param replies: The output of the Generator. A string.\n        :param pattern: The regular expression pattern to use to extract the answer text from the generator output.\n        '
        if pattern is None:
            return reply
        if (match := re.search(pattern, reply)):
            if not match.lastindex:
                return match.group(0)
            return match.group(1)
        return ''

    @staticmethod
    def _extract_reference_idxs(reply: str, reference_pattern: str) -> List[int]:
        if False:
            while True:
                i = 10
        document_idxs = re.findall(reference_pattern, reply)
        return [int(idx) - 1 for idx in document_idxs]

    @staticmethod
    def _check_num_groups_in_regex(pattern: str):
        if False:
            i = 10
            return i + 15
        num_groups = re.compile(pattern).groups
        if num_groups > 1:
            raise ValueError(f"Pattern '{pattern}' contains multiple capture groups. Please specify a pattern with at most one capture group.")