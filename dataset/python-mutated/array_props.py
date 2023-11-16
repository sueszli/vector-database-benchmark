"""
Example demonstrating partial mutations to array properties. The example
has two components, one of which has an list property which is mutated
incrementally. The other component replicates the list. In practice, the other
component would e.g. manage elements in the DOM, or other resources.
"""
from flexx import event

class Test1(event.Component):
    data = event.ListProp([], doc='An array property')

    @event.action
    def add(self, i):
        if False:
            while True:
                i = 10
        self._mutate_data([i], 'insert', len(self.data))

class Test2(event.Component):
    other = event.ComponentProp(None, settable=True)

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.data = []

    @event.reaction('other.data')
    def track_data(self, *events):
        if False:
            return 10
        for ev in events:
            if ev.mutation == 'set':
                self.data[:] = ev.new_value
            elif ev.mutation == 'insert':
                self.data[ev.index:ev.index] = ev.objects
            elif ev.mutation == 'remove':
                self.data[ev.index:ev.index + ev.objects] = []
            elif ev.mutation == 'replace':
                self.data[ev.index:ev.index + len(ev.objects)] = ev.objects
            else:
                raise NotImplementedError(ev.mutation)
test1 = Test1()
test2 = Test2(other=test1)
test1.add(4)
test1.add(7)
test1.add(6)
print(test2.data)
event.loop.iter()
print(test2.data)