from st2actions.runners.pythonrunner import Action
__all__ = ['GetLibraryPathAction']

class GetLibraryPathAction(Action):

    def run(self, module):
        if False:
            for i in range(10):
                print('nop')
        return __import__(module).__file__