from core.gui.base import DupeGuruGUIObject

class StatsLabel(DupeGuruGUIObject):

    def _view_updated(self):
        if False:
            return 10
        self.view.refresh()

    @property
    def display(self):
        if False:
            for i in range(10):
                print('nop')
        return self.app.stat_line

    def results_changed(self):
        if False:
            for i in range(10):
                print('nop')
        self.view.refresh()
    marking_changed = results_changed