from asciimatics.widgets import Frame, ListBox, Layout, Divider, Text, Button, TextBox, Widget
from asciimatics.scene import Scene
from asciimatics.screen import Screen
from asciimatics.exceptions import ResizeScreenError, NextScene, StopApplication
import sys

class ContactModel:

    def __init__(self):
        if False:
            print('Hello World!')
        self.current_id = None
        self.contacts = []

class ListView(Frame):

    def __init__(self, screen, model):
        if False:
            for i in range(10):
                print('nop')
        super(ListView, self).__init__(screen, screen.height * 2 // 3, screen.width * 2 // 3, on_load=self._reload_list, hover_focus=True, can_scroll=False, title='Contact List')
        self._model = model
        self._list_view = ListBox(Widget.FILL_FRAME, [(x['name'], i) for (i, x) in enumerate(self._model.contacts)], name='contacts', add_scroll_bar=True, on_change=self._on_pick, on_select=self._edit)
        self._edit_button = Button('Edit', self._edit)
        self._delete_button = Button('Delete', self._delete)
        layout = Layout([100], fill_frame=True)
        self.add_layout(layout)
        layout.add_widget(self._list_view)
        layout.add_widget(Divider())
        layout2 = Layout([1, 1, 1, 1])
        self.add_layout(layout2)
        layout2.add_widget(Button('Add', self._add), 0)
        layout2.add_widget(self._edit_button, 1)
        layout2.add_widget(self._delete_button, 2)
        layout2.add_widget(Button('Quit', self._quit), 3)
        self.fix()
        self._on_pick()

    def _on_pick(self):
        if False:
            i = 10
            return i + 15
        self._edit_button.disabled = self._list_view.value is None
        self._delete_button.disabled = self._list_view.value is None

    def _reload_list(self, new_value=None):
        if False:
            for i in range(10):
                print('nop')
        self._list_view.options = [(x['name'], i) for (i, x) in enumerate(self._model.contacts)]
        self._list_view.value = new_value

    def _add(self):
        if False:
            print('Hello World!')
        self._model.current_id = None
        raise NextScene('Edit Contact')

    def _edit(self):
        if False:
            return 10
        self.save()
        self._model.current_id = self.data['contacts']
        raise NextScene('Edit Contact')

    def _delete(self):
        if False:
            i = 10
            return i + 15
        self.save()
        del self._model.contacts[self.data['contacts']]
        self._reload_list()

    @staticmethod
    def _quit():
        if False:
            print('Hello World!')
        raise StopApplication('User pressed quit')

class ContactView(Frame):

    def __init__(self, screen, model):
        if False:
            while True:
                i = 10
        super(ContactView, self).__init__(screen, screen.height * 2 // 3, screen.width * 2 // 3, hover_focus=True, can_scroll=False, title='Contact Details', reduce_cpu=True)
        self._model = model
        layout = Layout([100], fill_frame=True)
        self.add_layout(layout)
        layout.add_widget(Text('Name:', 'name'))
        layout.add_widget(Text('Address:', 'address'))
        layout.add_widget(Text('Phone number:', 'phone'))
        layout.add_widget(Text('Email address:', 'email'))
        layout.add_widget(TextBox(Widget.FILL_FRAME, 'Notes:', 'notes', as_string=True, line_wrap=True))
        layout2 = Layout([1, 1, 1, 1])
        self.add_layout(layout2)
        layout2.add_widget(Button('OK', self._ok), 0)
        layout2.add_widget(Button('Cancel', self._cancel), 3)
        self.fix()

    def reset(self):
        if False:
            i = 10
            return i + 15
        super(ContactView, self).reset()
        if self._model.current_id is None:
            self.data = {'name': '', 'address': '', 'phone': '', 'email': '', 'notes': ''}
        else:
            self.data = self._model.contacts[self._model.current_id]

    def _ok(self):
        if False:
            for i in range(10):
                print('nop')
        self.save()
        if self._model.current_id is None:
            self._model.contacts.append(self.data)
        else:
            self._model.contacts[self._model.current_id] = self.data
        raise NextScene('Main')

    @staticmethod
    def _cancel():
        if False:
            i = 10
            return i + 15
        raise NextScene('Main')

def demo(screen, scene):
    if False:
        while True:
            i = 10
    scenes = [Scene([ListView(screen, contacts)], -1, name='Main'), Scene([ContactView(screen, contacts)], -1, name='Edit Contact')]
    screen.play(scenes, stop_on_resize=True, start_scene=scene, allow_int=True)
contacts = ContactModel()
last_scene = None
while True:
    try:
        Screen.wrapper(demo, catch_interrupt=True, arguments=[last_scene])
        sys.exit(0)
    except ResizeScreenError as e:
        last_scene = e.scene