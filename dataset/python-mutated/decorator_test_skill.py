from adapt.intent import IntentBuilder
from mycroft.skills.core import MycroftSkill
from mycroft.skills.core import intent_handler, intent_file_handler

class TestSkill(MycroftSkill):
    """ Test skill for intent_handler decorator. """

    @intent_handler(IntentBuilder('a').require('Keyword').build())
    def handler(self, message):
        if False:
            i = 10
            return i + 15
        pass

    @intent_file_handler('test.intent')
    def handler2(self, message):
        if False:
            return 10
        pass

    def stop(self):
        if False:
            while True:
                i = 10
        pass