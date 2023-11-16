"""UI implementations for use in testing.
"""
from bzrlib import progress, ui

class ProgressRecordingUIFactory(ui.UIFactory, progress.DummyProgress):
    """Captures progress updates made through it.
    
    This is overloaded as both the UIFactory and the progress model."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(ProgressRecordingUIFactory, self).__init__()
        self._calls = []
        self.depth = 0

    def nested_progress_bar(self):
        if False:
            return 10
        self.depth += 1
        return self

    def finished(self):
        if False:
            i = 10
            return i + 15
        self.depth -= 1

    def update(self, message, count=None, total=None):
        if False:
            while True:
                i = 10
        if self.depth == 1:
            self._calls.append(('update', count, total, message))