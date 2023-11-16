"""
Helper class for running scenarios at a command prompt. Asks questions, validates
and converts input, and returns answers.
"""

class Question:
    """
    A helper class to ask questions at a command prompt and validate and convert
    the answers.
    """

    def __init__(self, key, question, *validators):
        if False:
            i = 10
            return i + 15
        '\n        :param key: The key that is used for storing the answer in a dict, when\n                    multiple questions are asked in a set.\n        :param question: The question to ask.\n        :param validators: The answer is passed through the list of validators until\n                           one fails or they all pass. Validators may also convert the\n                           answer to another form, such as from a str to an int.\n        '
        self.key = key
        self.question = question
        self.validators = (Question.non_empty, *validators)

    @staticmethod
    def ask_questions(questions):
        if False:
            i = 10
            return i + 15
        '\n        Asks a set of questions and stores the answers in a dict.\n\n        :param questions: The list of questions to ask.\n        :return: A dict of answers.\n        '
        answers = {}
        for question in questions:
            answers[question.key] = Question.ask_question(question.question, *question.validators)
        return answers

    @staticmethod
    def ask_question(question, *validators):
        if False:
            while True:
                i = 10
        '\n        Asks a single question and validates it against a list of validators.\n        When an answer fails validation, the complaint is printed and the question\n        is asked again.\n\n        :param question: The question to ask.\n        :param validators: The list of validators that the answer must pass.\n        :return: The answer, converted to its final form by the validators.\n        '
        answer = None
        while answer is None:
            answer = input(question)
            for validator in validators:
                (answer, complaint) = validator(answer)
                if answer is None:
                    print(complaint)
                    break
        return answer

    @staticmethod
    def non_empty(answer):
        if False:
            return 10
        '\n        Validates that the answer is not empty.\n        :return: The non-empty answer, or None.\n        '
        return (answer if answer != '' else None, 'I need an answer. Please?')

    @staticmethod
    def is_yesno(answer):
        if False:
            return 10
        "\n        Validates a yes/no answer.\n        :return: True when the answer is 'y'; otherwise, False.\n        "
        return (answer.lower() == 'y', '')

    @staticmethod
    def is_int(answer):
        if False:
            print('Hello World!')
        '\n        Validates that the answer can be converted to an int.\n        :return: The int answer; otherwise, None.\n        '
        try:
            int_answer = int(answer)
        except ValueError:
            int_answer = None
        return (int_answer, f'{answer} must be a valid integer.')

    @staticmethod
    def is_letter(answer):
        if False:
            return 10
        '\n        Validates that the answer is a letter.\n        :return The letter answer, converted to uppercase; otherwise, None.\n        '
        return (answer.upper() if answer.isalpha() else None, f'{answer} must be a single letter.')

    @staticmethod
    def is_float(answer):
        if False:
            i = 10
            return i + 15
        '\n        Validate that the answer can be converted to a float.\n        :return The float answer; otherwise, None.\n        '
        try:
            float_answer = float(answer)
        except ValueError:
            float_answer = None
        return (float_answer, f'{answer} must be a valid float.')

    @staticmethod
    def in_range(lower, upper):
        if False:
            for i in range(10):
                print('nop')
        '\n        Validate that the answer is within a range. The answer must be of a type that can\n        be compared to the lower and upper bounds.\n        :return: The answer, if it is within the range; otherwise, None.\n        '

        def _validate(answer):
            if False:
                while True:
                    i = 10
            return (answer if lower <= answer <= upper else None, f'{answer} must be between {lower} and {upper}.')
        return _validate