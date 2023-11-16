import remi.gui as gui
from remi.gui import *
from remi import start, App

class screen1Widget(Container):

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(screen1Widget, self).__init__(**kwargs)
        self.style['position'] = 'absolute'
        self.style['overflow'] = 'auto'
        self.style['background-color'] = '#ffff80'
        self.style['left'] = '10px'
        self.style['top'] = '10px'
        self.style['margin'] = '0px'
        self.style['width'] = '427px'
        self.style['display'] = 'block'
        self.style['height'] = '480px'
        testlabel = Label('This is Screen 1!')
        mytextbox = TextInput(single_line=True, hint='Result Box. You can add content on screen 2')
        self.append(testlabel, 'testlabel')
        self.append(mytextbox, 'mytextbox')

class screen2Widget(Container):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(screen2Widget, self).__init__(**kwargs)
        self.style['position'] = 'absolute'
        self.style['overflow'] = 'auto'
        self.style['background-color'] = '#ffff80'
        self.style['left'] = '10px'
        self.style['top'] = '10px'
        self.style['margin'] = '0px'
        self.style['width'] = '427px'
        self.style['display'] = 'block'
        self.style['height'] = '480px'
        testlabel = Label('This is Screen 2!')
        mytextbox = TextInput(single_line=True, hint='Write something to be send to screen 1')
        btnsend = Button('Send Text Input to Screen 1')
        self.append(testlabel, 'testlabel')
        self.append(mytextbox, 'mytextbox')
        self.append(btnsend, 'btnsend')

class multiscreen(App):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if not 'editing_mode' in kwargs.keys():
            super(multiscreen, self).__init__(*args)

    def idle(self):
        if False:
            i = 10
            return i + 15
        pass

    def main(self):
        if False:
            while True:
                i = 10
        baseContainer = Container()
        baseContainer.attributes['class'] = 'Container  '
        baseContainer.attributes['editor_baseclass'] = 'Container'
        baseContainer.attributes['editor_varname'] = 'baseContainer'
        baseContainer.attributes['editor_tag_type'] = 'widget'
        baseContainer.attributes['editor_newclass'] = 'False'
        baseContainer.attributes['editor_constructor'] = '()'
        baseContainer.style['position'] = 'absolute'
        baseContainer.style['overflow'] = 'auto'
        baseContainer.style['left'] = '100px'
        baseContainer.style['top'] = '120px'
        baseContainer.style['margin'] = '0px'
        baseContainer.style['border-style'] = 'solid'
        baseContainer.style['width'] = '670px'
        baseContainer.style['display'] = 'block'
        baseContainer.style['border-width'] = '1px'
        baseContainer.style['height'] = '550px'
        menuContainer = Container()
        menuContainer.attributes['class'] = 'Container'
        menuContainer.attributes['editor_baseclass'] = 'Container'
        menuContainer.attributes['editor_varname'] = 'menuContainer'
        menuContainer.attributes['editor_tag_type'] = 'widget'
        menuContainer.attributes['editor_newclass'] = 'False'
        menuContainer.attributes['editor_constructor'] = '()'
        menuContainer.style['position'] = 'absolute'
        menuContainer.style['overflow'] = 'auto'
        menuContainer.style['left'] = '10px'
        menuContainer.style['top'] = '10px'
        menuContainer.style['margin'] = '0px'
        menuContainer.style['border-style'] = 'solid'
        menuContainer.style['width'] = '180px'
        menuContainer.style['display'] = 'block'
        menuContainer.style['border-width'] = '1px'
        menuContainer.style['height'] = '500px'
        btnScreen2 = Button('Screen 2')
        btnScreen2.attributes['class'] = 'Button'
        btnScreen2.attributes['editor_baseclass'] = 'Button'
        btnScreen2.attributes['editor_varname'] = 'btnScreen2'
        btnScreen2.attributes['editor_tag_type'] = 'widget'
        btnScreen2.attributes['editor_newclass'] = 'False'
        btnScreen2.attributes['editor_constructor'] = "('Screen 2')"
        btnScreen2.style['position'] = 'absolute'
        btnScreen2.style['overflow'] = 'auto'
        btnScreen2.style['left'] = '5px'
        btnScreen2.style['top'] = '60px'
        btnScreen2.style['margin'] = '0px'
        btnScreen2.style['width'] = '150px'
        btnScreen2.style['display'] = 'block'
        btnScreen2.style['height'] = '30px'
        menuContainer.append(btnScreen2, 'btnScreen2')
        btnScreen1 = Button('Screen 1')
        btnScreen1.attributes['class'] = 'Button'
        btnScreen1.attributes['editor_baseclass'] = 'Button'
        btnScreen1.attributes['editor_varname'] = 'btnScreen1'
        btnScreen1.attributes['editor_tag_type'] = 'widget'
        btnScreen1.attributes['editor_newclass'] = 'False'
        btnScreen1.attributes['editor_constructor'] = "('Screen 1')"
        btnScreen1.style['position'] = 'absolute'
        btnScreen1.style['overflow'] = 'auto'
        btnScreen1.style['left'] = '5px'
        btnScreen1.style['top'] = '10px'
        btnScreen1.style['margin'] = '0px'
        btnScreen1.style['width'] = '150px'
        btnScreen1.style['display'] = 'block'
        btnScreen1.style['height'] = '30px'
        menuContainer.append(btnScreen1, 'btnScreen1')
        baseContainer.append(menuContainer, 'menuContainer')
        baseContainer.children['menuContainer'].children['btnScreen2'].onclick.do(self.onclick_btnScreen2)
        baseContainer.children['menuContainer'].children['btnScreen1'].onclick.do(self.onclick_btnScreen1)
        contentContainer = Container()
        contentContainer.attributes['class'] = 'Container'
        contentContainer.attributes['editor_baseclass'] = 'Container'
        contentContainer.attributes['editor_varname'] = 'contentContainer'
        contentContainer.attributes['editor_tag_type'] = 'widget'
        contentContainer.attributes['editor_newclass'] = 'False'
        contentContainer.attributes['editor_constructor'] = '()'
        contentContainer.style['position'] = 'absolute'
        contentContainer.style['overflow'] = 'auto'
        contentContainer.style['left'] = '200px'
        contentContainer.style['top'] = '10px'
        contentContainer.style['margin'] = '0px'
        contentContainer.style['border-style'] = 'solid'
        contentContainer.style['width'] = '450px'
        contentContainer.style['display'] = 'block'
        contentContainer.style['border-width'] = '1px'
        contentContainer.style['height'] = '500px'
        self.screen1 = screen1Widget()
        self.screen2 = screen2Widget()
        contentContainer.append(self.screen1, 'screen1')
        self.screen2.children['btnsend'].onclick.do(self.send_text_to_screen1)
        baseContainer.append(contentContainer, 'contentContainer')
        self.baseContainer = baseContainer
        return self.baseContainer

    def onclick_btnScreen2(self, emitter):
        if False:
            return 10
        if 'screen1' in self.baseContainer.children['contentContainer'].children.keys():
            self.baseContainer.children['contentContainer'].remove_child(self.baseContainer.children['contentContainer'].children['screen1'])
        self.baseContainer.children['contentContainer'].append(self.screen2, 'screen2')

    def onclick_btnScreen1(self, emitter):
        if False:
            print('Hello World!')
        if 'screen2' in self.baseContainer.children['contentContainer'].children.keys():
            self.baseContainer.children['contentContainer'].remove_child(self.baseContainer.children['contentContainer'].children['screen2'])
        self.baseContainer.children['contentContainer'].append(self.screen1, 'screen1')

    def send_text_to_screen1(self, emitter):
        if False:
            return 10
        self.screen1.children['mytextbox'].set_text(self.screen2.children['mytextbox'].get_text())
configuration = {'config_port': 8081, 'config_address': '0.0.0.0', 'config_resourcepath': './res/', 'config_enable_file_cache': True, 'config_multiple_instance': True, 'config_project_name': 'multiscreen', 'config_start_browser': True}
if __name__ == '__main__':
    start(multiscreen, address=configuration['config_address'], port=configuration['config_port'], multiple_instance=configuration['config_multiple_instance'], enable_file_cache=configuration['config_enable_file_cache'], start_browser=configuration['config_start_browser'])