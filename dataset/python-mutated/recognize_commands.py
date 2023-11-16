"""Stream accuracy recognize commands."""
import collections
import numpy as np

class RecognizeResult(object):
    """Save recognition result temporarily.

  Attributes:
    founded_command: A string indicating the word just founded. Default value
      is '_silence_'
    score: A float representing the confidence of founded word. Default
      value is zero.
    is_new_command: A boolean indicating if the founded command is a new one
      against the last one. Default value is False.
  """

    def __init__(self, founded_command='_silence_', score=0.0, is_new_command=False):
        if False:
            while True:
                i = 10
        'Construct a recognition result.\n\n    Args:\n      founded_command: A string indicating the word just founded.\n      score: A float representing the confidence of founded word.\n      is_new_command: A boolean indicating if the founded command is a new one\n        against the last one.\n    '
        self._founded_command = founded_command
        self._score = score
        self._is_new_command = is_new_command

    @property
    def founded_command(self):
        if False:
            for i in range(10):
                print('nop')
        return self._founded_command

    @founded_command.setter
    def founded_command(self, value):
        if False:
            i = 10
            return i + 15
        self._founded_command = value

    @property
    def score(self):
        if False:
            print('Hello World!')
        return self._score

    @score.setter
    def score(self, value):
        if False:
            return 10
        self._score = value

    @property
    def is_new_command(self):
        if False:
            while True:
                i = 10
        return self._is_new_command

    @is_new_command.setter
    def is_new_command(self, value):
        if False:
            i = 10
            return i + 15
        self._is_new_command = value

class RecognizeCommands(object):
    """Smooth the inference results by using average window.

  Maintain a slide window over the audio stream, which adds new result(a pair of
  the 1.confidences of all classes and 2.the start timestamp of input audio
  clip) directly the inference produces one and removes the most previous one
  and other abnormal values. Then it smooth the results in the window to get
  the most reliable command in this period.

  Attributes:
    _label: A list containing commands at corresponding lines.
    _average_window_duration: The length of average window.
    _detection_threshold: A confidence threshold for filtering out unreliable
      command.
    _suppression_ms: Milliseconds every two reliable founded commands should
      apart.
    _minimum_count: An integer count indicating the minimum results the average
      window should cover.
    _previous_results: A deque to store previous results.
    _label_count: The length of label list.
    _previous_top_label: Last founded command. Initial value is '_silence_'.
    _previous_top_time: The timestamp of _previous results. Default is -np.inf.
  """

    def __init__(self, labels, average_window_duration_ms, detection_threshold, suppression_ms, minimum_count):
        if False:
            while True:
                i = 10
        'Init the RecognizeCommands with parameters used for smoothing.'
        self._labels = labels
        self._average_window_duration_ms = average_window_duration_ms
        self._detection_threshold = detection_threshold
        self._suppression_ms = suppression_ms
        self._minimum_count = minimum_count
        self._previous_results = collections.deque()
        self._label_count = len(labels)
        self._previous_top_label = '_silence_'
        self._previous_top_time = -np.inf

    def process_latest_result(self, latest_results, current_time_ms, recognize_element):
        if False:
            print('Hello World!')
        "Smoothing the results in average window when a new result is added in.\n\n    Receive a new result from inference and put the founded command into\n    a RecognizeResult instance after the smoothing procedure.\n\n    Args:\n      latest_results: A list containing the confidences of all labels.\n      current_time_ms: The start timestamp of the input audio clip.\n      recognize_element: An instance of RecognizeResult to store founded\n        command, its scores and if it is a new command.\n\n    Raises:\n      ValueError: The length of this result from inference doesn't match\n        label count.\n      ValueError: The timestamp of this result is earlier than the most\n        previous one in the average window\n    "
        if latest_results.shape[0] != self._label_count:
            raise ValueError('The results for recognition should contain {} elements, but there are {} produced'.format(self._label_count, latest_results.shape[0]))
        if self._previous_results.__len__() != 0 and current_time_ms < self._previous_results[0][0]:
            raise ValueError('Results must be fed in increasing time order, but receive a timestamp of {}, which was earlier than the previous one of {}'.format(current_time_ms, self._previous_results[0][0]))
        self._previous_results.append([current_time_ms, latest_results])
        time_limit = current_time_ms - self._average_window_duration_ms
        while time_limit > self._previous_results[0][0]:
            self._previous_results.popleft()
        how_many_results = self._previous_results.__len__()
        earliest_time = self._previous_results[0][0]
        sample_duration = current_time_ms - earliest_time
        if how_many_results < self._minimum_count or sample_duration < self._average_window_duration_ms / 4:
            recognize_element.founded_command = self._previous_top_label
            recognize_element.score = 0.0
            recognize_element.is_new_command = False
            return
        average_scores = np.zeros(self._label_count)
        for item in self._previous_results:
            score = item[1]
            for i in range(score.size):
                average_scores[i] += score[i] / how_many_results
        sorted_averaged_index_score = []
        for i in range(self._label_count):
            sorted_averaged_index_score.append([i, average_scores[i]])
        sorted_averaged_index_score = sorted(sorted_averaged_index_score, key=lambda p: p[1], reverse=True)
        current_top_index = sorted_averaged_index_score[0][0]
        current_top_label = self._labels[current_top_index]
        current_top_score = sorted_averaged_index_score[0][1]
        time_since_last_top = 0
        if self._previous_top_label == '_silence_' or self._previous_top_time == -np.inf:
            time_since_last_top = np.inf
        else:
            time_since_last_top = current_time_ms - self._previous_top_time
        if current_top_score > self._detection_threshold and current_top_label != self._previous_top_label and (time_since_last_top > self._suppression_ms):
            self._previous_top_label = current_top_label
            self._previous_top_time = current_time_ms
            recognize_element.is_new_command = True
        else:
            recognize_element.is_new_command = False
        recognize_element.founded_command = current_top_label
        recognize_element.score = current_top_score