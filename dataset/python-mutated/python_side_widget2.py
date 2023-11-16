from flexx import flx

class UserInput(flx.PyWidget):

    def init(self):
        if False:
            while True:
                i = 10
        with flx.VBox():
            self.edit = flx.LineEdit(placeholder_text='Your name')
            flx.Widget(flex=1)

    @flx.reaction('edit.user_done')
    def update_user(self, *events):
        if False:
            i = 10
            return i + 15
        new_text = self.root.store.username + '\n' + self.edit.text
        self.root.store.set_username(new_text)
        self.edit.set_text('')

class SomeInfoWidget(flx.PyWidget):

    def init(self):
        if False:
            print('Hello World!')
        with flx.FormLayout():
            self.label = flx.Label(title='name:')
            flx.Widget(flex=1)

    @flx.reaction
    def update_label(self):
        if False:
            for i in range(10):
                print('nop')
        self.label.set_text(self.root.store.username)

class Store(flx.PyComponent):
    username = flx.StringProp(settable=True)

class Example(flx.PyWidget):
    store = flx.ComponentProp()

    def init(self):
        if False:
            while True:
                i = 10
        self._mutate_store(Store())
        with flx.HSplit():
            UserInput()
            flx.Widget(style='background:#eee;')
            SomeInfoWidget()
if __name__ == '__main__':
    m = flx.launch(Example, 'default-browser', backend='flask')
    flx.run()