"""
This example demonstrates how a property can be set to follow other properties
by setting the property to be a function, which is converted to a reaction by
Flexx.

This feature makes basic plumbing very easy, like e.g. showing values
of properties in a label widget.
"""
from flexx import event

class Person(event.Component):
    first_name = event.StringProp('Jane', settable=True)
    last_name = event.StringProp('Doe', settable=True)

class Greeter(event.Component):
    message = event.StringProp('', settable=True)

    @event.reaction
    def show_message(self):
        if False:
            for i in range(10):
                print('nop')
        print('Message:', self.message)
p = Person()
g = Greeter(message=lambda : p.first_name + ' ' + p.last_name)
p.set_first_name('Alice')
event.loop.iter()