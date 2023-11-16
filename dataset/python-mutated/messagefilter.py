from robot.output.loggerhelper import IsLogged
from robot.model import SuiteVisitor

class MessageFilter(SuiteVisitor):

    def __init__(self, log_level=None):
        if False:
            return 10
        self.is_logged = IsLogged(log_level or 'TRACE')

    def start_suite(self, suite):
        if False:
            return 10
        if self.is_logged.level == 'TRACE':
            return False

    def start_keyword(self, keyword):
        if False:
            while True:
                i = 10
        for item in list(keyword.body):
            if item.type == item.MESSAGE and (not self.is_logged(item.level)):
                keyword.body.remove(item)