from typing import List, Any, Optional, Tuple, Union, Dict
import logging
from abc import ABC
logger = logging.getLogger(__name__)

class Pred(ABC):
    """
    Abstract base class for predictions of every task
    """

    def __init__(self, id: str, prediction: List[Any], context: str):
        if False:
            while True:
                i = 10
        self.id = id
        self.prediction = prediction
        self.context = context

    def to_json(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

class QACandidate:
    """
    A single QA candidate answer.
    """

    def __init__(self, answer_type: str, score: float, offset_answer_start: int, offset_answer_end: int, offset_unit: str, aggregation_level: str, probability: Optional[float]=None, n_passages_in_doc: Optional[int]=None, passage_id: Optional[str]=None, confidence: Optional[float]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param answer_type: The category that this answer falls into e.g. "no_answer", "yes", "no" or "span"\n        :param score: The score representing the model\'s confidence of this answer\n        :param offset_answer_start: The index of the start of the answer span (whether it is char or tok is stated in self.offset_unit)\n        :param offset_answer_end: The index of the start of the answer span (whether it is char or tok is stated in self.offset_unit)\n        :param offset_unit: States whether the offsets refer to character or token indices\n        :param aggregation_level: States whether this candidate and its indices are on a passage level (pre aggregation) or on a document level (post aggregation)\n        :param probability: The probability the model assigns to the answer\n        :param n_passages_in_doc: Number of passages that make up the document\n        :param passage_id: The id of the passage which contains this candidate answer\n        :param confidence: The (calibrated) confidence score representing the model\'s predicted accuracy of the index of the start of the answer span\n        '
        self.answer_type = answer_type
        self.score = score
        self.probability = probability
        self.answer = None
        self.offset_answer_start = offset_answer_start
        self.offset_answer_end = offset_answer_end
        self.answer_support = None
        self.offset_answer_support_start = None
        self.offset_answer_support_end = None
        self.context_window = None
        self.offset_context_window_start = None
        self.offset_context_window_end = None
        self.offset_unit = offset_unit
        self.aggregation_level = aggregation_level
        self.n_passages_in_doc = n_passages_in_doc
        self.passage_id = passage_id
        self.confidence = confidence
        self.meta = None

    def set_context_window(self, context_window_size: int, clear_text: str):
        if False:
            for i in range(10):
                print('nop')
        (window_str, start_ch, end_ch) = self._create_context_window(context_window_size, clear_text)
        self.context_window = window_str
        self.offset_context_window_start = start_ch
        self.offset_context_window_end = end_ch

    def set_answer_string(self, token_offsets: List[int], document_text: str):
        if False:
            print('Hello World!')
        (pred_str, self.offset_answer_start, self.offset_answer_end) = self._span_to_string(token_offsets, document_text)
        self.offset_unit = 'char'
        self._add_answer(pred_str)

    def _add_answer(self, string: str):
        if False:
            i = 10
            return i + 15
        '\n        Set the answer string. This method will check that the answer given is valid given the start\n        and end indices that are stored in the object.\n        '
        if string == '':
            self.answer = 'no_answer'
            if self.offset_answer_start != 0 or self.offset_answer_end != 0:
                logger.error('Both start and end offsets should be 0: \n%s, %s with a no_answer. ', self.offset_answer_start, self.offset_answer_end)
        else:
            self.answer = string
            if self.offset_answer_end - self.offset_answer_start <= 0:
                logger.error('End offset comes before start offset: \n(%s, %s) with a span answer. ', self.offset_answer_start, self.offset_answer_end)
            elif self.offset_answer_end <= 0:
                logger.error('Invalid end offset: \n(%s, %s) with a span answer. ', self.offset_answer_start, self.offset_answer_end)

    def _create_context_window(self, context_window_size: int, clear_text: str) -> Tuple[str, int, int]:
        if False:
            return 10
        '\n        Extract from the clear_text a window that contains the answer and (usually) some amount of text on either\n        side of the answer. Useful for cases where the answer and its surrounding context needs to be\n        displayed in a UI. If the self.context_window_size is smaller than the extracted answer, it will be\n        enlarged so that it can contain the answer\n\n        :param context_window_size: The size of the context window to be generated. Note that the window size may be increased if the answer is longer.\n        :param clear_text: The text from which the answer is extracted\n        '
        if self.offset_answer_start == 0 and self.offset_answer_end == 0:
            return ('', 0, 0)
        else:
            len_ans = self.offset_answer_end - self.offset_answer_start
            context_window_size = max(context_window_size, len_ans + 1)
            len_text = len(clear_text)
            midpoint = int(len_ans / 2) + self.offset_answer_start
            half_window = int(context_window_size / 2)
            window_start_ch = midpoint - half_window
            window_end_ch = midpoint + half_window
            overhang_start = max(0, -window_start_ch)
            overhang_end = max(0, window_end_ch - len_text)
            window_start_ch -= overhang_end
            window_start_ch = max(0, window_start_ch)
            window_end_ch += overhang_start
            window_end_ch = min(len_text, window_end_ch)
        window_str = clear_text[window_start_ch:window_end_ch]
        return (window_str, window_start_ch, window_end_ch)

    def _span_to_string(self, token_offsets: List[int], clear_text: str) -> Tuple[str, int, int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Generates a string answer span using self.offset_answer_start and self.offset_answer_end. If the candidate\n        is a no answer, an empty string is returned\n\n        :param token_offsets: A list of ints which give the start character index of the corresponding token\n        :param clear_text: The text from which the answer span is to be extracted\n        :return: The string answer span, followed by the start and end character indices\n        '
        if self.offset_unit != 'token':
            logger.error('QACandidate needs to have self.offset_unit=token before calling _span_to_string() (id = %s)', self.passage_id)
        start_t = self.offset_answer_start
        end_t = self.offset_answer_end
        if start_t == -1 and end_t == -1:
            return ('', 0, 0)
        n_tokens = len(token_offsets)
        end_t += 1
        end_t = min(end_t, n_tokens)
        start_ch = int(token_offsets[start_t])
        if end_t == n_tokens:
            end_ch = len(clear_text)
        else:
            end_ch = token_offsets[end_t]
        final_text = clear_text[start_ch:end_ch]
        cleaned_final_text = final_text.strip()
        if not cleaned_final_text:
            self.answer_type = 'no_answer'
            return ('', 0, 0)
        left_offset = len(final_text) - len(final_text.lstrip())
        if left_offset:
            start_ch = start_ch + left_offset
        end_ch = start_ch + len(cleaned_final_text)
        return (cleaned_final_text, start_ch, end_ch)

    def to_doc_level(self, start: int, end: int):
        if False:
            i = 10
            return i + 15
        "\n        Populate the start and end indices with document level indices. Changes aggregation level to 'document'\n        "
        self.offset_answer_start = start
        self.offset_answer_end = end
        self.aggregation_level = 'document'

    def to_list(self) -> List[Optional[Union[str, int, float]]]:
        if False:
            while True:
                i = 10
        return [self.answer, self.offset_answer_start, self.offset_answer_end, self.score, self.passage_id]

class QAPred(Pred):
    """
    A set of QA predictions for a passage or a document. The candidates are stored in QAPred.prediction which is a
    list of QACandidate objects. Also contains all attributes needed to convert the object into json format and also
    to create a context window for a UI
    """

    def __init__(self, id: str, prediction: List[QACandidate], context: str, question: str, token_offsets: List[int], context_window_size: int, aggregation_level: str, no_answer_gap: float, ground_truth_answer: Optional[str]=None, answer_types: Optional[List[str]]=None):
        if False:
            i = 10
            return i + 15
        '\n        :param id: The id of the passage or document\n        :param prediction: A list of QACandidate objects for the given question and document\n        :param context: The text passage from which the answer can be extracted\n        :param question: The question being posed\n        :param token_offsets: A list of ints indicating the start char index of each token\n        :param context_window_size: The number of chars in the text window around the answer\n        :param aggregation_level: States whether this candidate and its indices are on a passage level (pre aggregation) or on a document level (post aggregation)\n        :param no_answer_gap: How much the QuestionAnsweringHead.no_ans_boost needs to change to turn a no_answer to a positive answer\n        :param ground_truth_answer: Ground truth answers\n        :param answer_types: List of answer_types supported by this task e.g. ["span", "yes_no", "no_answer"]\n        '
        if answer_types is None:
            answer_types = []
        super().__init__(id, prediction, context)
        self.question = question
        self.token_offsets = token_offsets
        self.context_window_size = context_window_size
        self.aggregation_level = aggregation_level
        self.answer_types = answer_types
        self.ground_truth_answer = ground_truth_answer
        self.no_answer_gap = no_answer_gap
        self.n_passages = self.prediction[0].n_passages_in_doc
        for qa_candidate in self.prediction:
            qa_candidate.set_answer_string(token_offsets, self.context)
            qa_candidate.set_context_window(self.context_window_size, self.context)

    def to_json(self, squad=False) -> Dict:
        if False:
            return 10
        '\n        Converts the information stored in the object into a json format.\n\n        :param squad: If True, no_answers are represented by the empty string instead of "no_answer"\n        '
        answers = self._answers_to_json(self.id, squad)
        ret = {'task': 'qa', 'predictions': [{'question': self.question, 'id': self.id, 'ground_truth': self.ground_truth_answer, 'answers': answers, 'no_ans_gap': self.no_answer_gap}]}
        if squad:
            del ret['predictions'][0]['id']
            ret['predictions'][0]['question_id'] = self.id
        return ret

    def _answers_to_json(self, ext_id, squad=False) -> List[Dict]:
        if False:
            i = 10
            return i + 15
        '\n        Convert all answers into a json format.\n\n        :param ext_id: ID of the question document pair.\n        :param squad: If True, no_answers are represented by the empty string instead of "no_answer".\n        '
        ret = []
        for qa_candidate in self.prediction:
            if squad and qa_candidate.answer == 'no_answer':
                answer_string = ''
            else:
                answer_string = qa_candidate.answer
            curr = {'score': qa_candidate.score, 'probability': None, 'answer': answer_string, 'offset_answer_start': qa_candidate.offset_answer_start, 'offset_answer_end': qa_candidate.offset_answer_end, 'context': qa_candidate.context_window, 'offset_context_start': qa_candidate.offset_context_window_start, 'offset_context_end': qa_candidate.offset_context_window_end, 'document_id': ext_id}
            ret.append(curr)
        return ret

    def to_squad_eval(self) -> Dict:
        if False:
            return 10
        return self.to_json(squad=True)