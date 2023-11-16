"""As a Python Fire demo, a Collector collects widgets, and nobody knows why."""
import fire
from examples.widget import widget

class Collector(object):
    """A Collector has one Widget, but wants more."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.widget = widget.Widget()
        self.desired_widget_count = 10

    def collect_widgets(self):
        if False:
            print('Hello World!')
        'Returns all the widgets the Collector wants.'
        return [widget.Widget() for _ in range(self.desired_widget_count)]

def main():
    if False:
        return 10
    fire.Fire(Collector(), name='collector')
if __name__ == '__main__':
    main()