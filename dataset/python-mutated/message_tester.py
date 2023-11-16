"""
    This message tester lets a user input a Message event and a json test
    case, and allows the evaluation of the Message event based on the
    test case

    It is supposed to be run in the Mycroft virtualenv, and python-tk
    must be installed (on Ubuntu: apt-get install python-tk)
"""
from tkinter import Label, Button, Tk, NORMAL, END, DISABLED
from tkinter.scrolledtext import ScrolledText
import skill_tester
import ast
EXAMPLE_EVENT = "{\n  'expect_response': False,\n  'utterance': 'Recording audio for 600 seconds'\n}"
EXAMPLE_TEST_CASE = '{\n  "utterance": "record",\n  "intent_type": "AudioRecordSkillIntent",\n  "intent": {\n    "AudioRecordSkillKeyword": "record"\n  },\n  "expected_response": ".*(recording|audio)"\n}'

class MessageTester:

    def __init__(self, root):
        if False:
            i = 10
            return i + 15
        root.title('Message tester')
        Label(root, text='Enter message event below', bg='light green').pack()
        self.event_field = ScrolledText(root, width=180, height=10)
        self.event_field.pack()
        Label(root, text='Enter test case below', bg='light green').pack()
        self.test_case_field = ScrolledText(root, width=180, height=20)
        self.test_case_field.pack()
        Label(root, text='Test result:', bg='light green').pack()
        self.result_field = ScrolledText(root, width=180, height=10)
        self.result_field.pack()
        self.result_field.config(state=DISABLED)
        self.button = Button(root, text='Evaluate', fg='red', command=self._clicked)
        self.button.pack()
        self.event_field.delete('1.0', END)
        self.event_field.insert('insert', EXAMPLE_EVENT)
        self.test_case_field.delete('1.0', END)
        self.test_case_field.insert('insert', EXAMPLE_TEST_CASE)

    def _clicked(self):
        if False:
            return 10
        event = self.event_field.get('1.0', END)
        test_case = self.test_case_field.get('1.0', END)
        evaluation = skill_tester.EvaluationRule(ast.literal_eval(test_case))
        evaluation.evaluate(ast.literal_eval(event))
        self.result_field.config(state=NORMAL)
        self.result_field.delete('1.0', END)
        self.result_field.insert('insert', evaluation.rule)
        self.result_field.config(state=DISABLED)
r = Tk()
app = MessageTester(r)
r.mainloop()