from datetime import datetime
from chatterbot.logic import LogicAdapter
from chatterbot.conversation import Statement
from chatterbot.exceptions import OptionalDependencyImportError

class TimeLogicAdapter(LogicAdapter):
    """
    The TimeLogicAdapter returns the current time.

    :kwargs:
        * *positive* (``list``) --
          The time-related questions used to identify time questions.
          Defaults to a list of English sentences.
        * *negative* (``list``) --
          The non-time-related questions used to identify time questions.
          Defaults to a list of English sentences.
    """

    def __init__(self, chatbot, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(chatbot, **kwargs)
        try:
            from nltk import NaiveBayesClassifier
        except ImportError:
            message = 'Unable to import "nltk".\nPlease install "nltk" before using the TimeLogicAdapter:\npip3 install nltk'
            raise OptionalDependencyImportError(message)
        self.positive = kwargs.get('positive', ['what time is it', 'hey what time is it', 'do you have the time', 'do you know the time', 'do you know what time it is', 'what is the time'])
        self.negative = kwargs.get('negative', ['it is time to go to sleep', 'what is your favorite color', 'i had a great time', 'thyme is my favorite herb', 'do you have time to look at my essay', 'how do you have the time to do all thiswhat is it'])
        labeled_data = [(name, 0) for name in self.negative] + [(name, 1) for name in self.positive]
        train_set = [(self.time_question_features(text), n) for (text, n) in labeled_data]
        self.classifier = NaiveBayesClassifier.train(train_set)

    def time_question_features(self, text):
        if False:
            return 10
        '\n        Provide an analysis of significant features in the string.\n        '
        features = {}
        all_words = ' '.join(self.positive + self.negative).split()
        all_first_words = []
        for sentence in self.positive + self.negative:
            all_first_words.append(sentence.split(' ', 1)[0])
        for word in text.split():
            features['first_word({})'.format(word)] = word in all_first_words
        for word in text.split():
            features['contains({})'.format(word)] = word in all_words
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            features['count({})'.format(letter)] = text.lower().count(letter)
            features['has({})'.format(letter)] = letter in text.lower()
        return features

    def process(self, statement, additional_response_selection_parameters=None):
        if False:
            print('Hello World!')
        now = datetime.now()
        time_features = self.time_question_features(statement.text.lower())
        confidence = self.classifier.classify(time_features)
        response = Statement(text='The current time is ' + now.strftime('%I:%M %p'))
        response.confidence = confidence
        return response