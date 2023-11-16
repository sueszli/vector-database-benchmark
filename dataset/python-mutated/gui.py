""" Interface for interacting with the Mycroft gui qml viewer. """
from os.path import join
from mycroft.configuration import Configuration
from mycroft.messagebus.message import Message
from mycroft.util import resolve_resource_file

class SkillGUI:
    """SkillGUI - Interface to the Graphical User Interface

    Values set in this class are synced to the GUI, accessible within QML
    via the built-in sessionData mechanism.  For example, in Python you can
    write in a skill:
        self.gui['temp'] = 33
        self.gui.show_page('Weather.qml')
    Then in the Weather.qml you'd access the temp via code such as:
        text: sessionData.time
    """

    def __init__(self, skill):
        if False:
            return 10
        self.__session_data = {}
        self.page = None
        self.skill = skill
        self.on_gui_changed_callback = None
        self.config = Configuration.get()

    @property
    def connected(self):
        if False:
            while True:
                i = 10
        'Returns True if at least 1 gui is connected, else False'
        if self.skill.bus:
            reply = self.skill.bus.wait_for_response(Message('gui.status.request'), 'gui.status.request.response')
            if reply:
                return reply.data['connected']
        return False

    @property
    def remote_url(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns configuration value for url of remote-server.'
        return self.config.get('remote-server')

    def build_message_type(self, event):
        if False:
            print('Hello World!')
        'Builds a message matching the output from the enclosure.'
        return '{}.{}'.format(self.skill.skill_id, event)

    def setup_default_handlers(self):
        if False:
            i = 10
            return i + 15
        'Sets the handlers for the default messages.'
        msg_type = self.build_message_type('set')
        self.skill.add_event(msg_type, self.gui_set)

    def register_handler(self, event, handler):
        if False:
            while True:
                i = 10
        'Register a handler for GUI events.\n\n        When using the triggerEvent method from Qt\n        triggerEvent("event", {"data": "cool"})\n\n        Args:\n            event (str):    event to catch\n            handler:        function to handle the event\n        '
        msg_type = self.build_message_type(event)
        self.skill.add_event(msg_type, handler)

    def set_on_gui_changed(self, callback):
        if False:
            print('Hello World!')
        'Registers a callback function to run when a value is\n        changed from the GUI.\n\n        Args:\n            callback:   Function to call when a value is changed\n        '
        self.on_gui_changed_callback = callback

    def gui_set(self, message):
        if False:
            i = 10
            return i + 15
        'Handler catching variable changes from the GUI.\n\n        Args:\n            message: Messagebus message\n        '
        for key in message.data:
            self[key] = message.data[key]
        if self.on_gui_changed_callback:
            self.on_gui_changed_callback()

    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        'Implements set part of dict-like behaviour with named keys.'
        self.__session_data[key] = value
        if self.page:
            data = self.__session_data.copy()
            data.update({'__from': self.skill.skill_id})
            self.skill.bus.emit(Message('gui.value.set', data))

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        'Implements get part of dict-like behaviour with named keys.'
        return self.__session_data[key]

    def get(self, *args, **kwargs):
        if False:
            print('Hello World!')
        'Implements the get method for accessing dict keys.'
        return self.__session_data.get(*args, **kwargs)

    def __contains__(self, key):
        if False:
            while True:
                i = 10
        'Implements the "in" operation.'
        return self.__session_data.__contains__(key)

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        'Reset the value dictionary, and remove namespace from GUI.\n\n        This method does not close the GUI for a Skill. For this purpose see\n        the `release` method.\n        '
        self.__session_data = {}
        self.page = None
        self.skill.bus.emit(Message('gui.clear.namespace', {'__from': self.skill.skill_id}))

    def send_event(self, event_name, params=None):
        if False:
            print('Hello World!')
        'Trigger a gui event.\n\n        Args:\n            event_name (str): name of event to be triggered\n            params: json serializable object containing any parameters that\n                    should be sent along with the request.\n        '
        params = params or {}
        self.skill.bus.emit(Message('gui.event.send', {'__from': self.skill.skill_id, 'event_name': event_name, 'params': params}))

    def show_page(self, name, override_idle=None, override_animations=False):
        if False:
            while True:
                i = 10
        'Begin showing the page in the GUI\n\n        Args:\n            name (str): Name of page (e.g "mypage.qml") to display\n            override_idle (boolean, int):\n                True: Takes over the resting page indefinitely\n                (int): Delays resting page for the specified number of\n                       seconds.\n            override_animations (boolean):\n                True: Disables showing all platform skill animations.\n                False: \'Default\' always show animations.\n        '
        self.show_pages([name], 0, override_idle, override_animations)

    def show_pages(self, page_names, index=0, override_idle=None, override_animations=False):
        if False:
            print('Hello World!')
        'Begin showing the list of pages in the GUI.\n\n        Args:\n            page_names (list): List of page names (str) to display, such as\n                               ["Weather.qml", "Forecast.qml", "Details.qml"]\n            index (int): Page number (0-based) to show initially.  For the\n                         above list a value of 1 would start on "Forecast.qml"\n            override_idle (boolean, int):\n                True: Takes over the resting page indefinitely\n                (int): Delays resting page for the specified number of\n                       seconds.\n            override_animations (boolean):\n                True: Disables showing all platform skill animations.\n                False: \'Default\' always show animations.\n        '
        if not isinstance(page_names, list):
            raise ValueError('page_names must be a list')
        if index > len(page_names):
            raise ValueError('Default index is larger than page list length')
        self.page = page_names[index]
        data = self.__session_data.copy()
        data.update({'__from': self.skill.skill_id})
        self.skill.bus.emit(Message('gui.value.set', data))
        page_urls = []
        for name in page_names:
            if name.startswith('SYSTEM'):
                page = resolve_resource_file(join('ui', name))
            else:
                page = self.skill.find_resource(name, 'ui')
            if page:
                if self.config.get('remote'):
                    page_urls.append(self.remote_url + '/' + page)
                else:
                    page_urls.append('file://' + page)
            else:
                raise FileNotFoundError('Unable to find page: {}'.format(name))
        self.skill.bus.emit(Message('gui.page.show', {'page': page_urls, 'index': index, '__from': self.skill.skill_id, '__idle': override_idle, '__animations': override_animations}))

    def remove_page(self, page):
        if False:
            while True:
                i = 10
        'Remove a single page from the GUI.\n\n        Args:\n            page (str): Page to remove from the GUI\n        '
        return self.remove_pages([page])

    def remove_pages(self, page_names):
        if False:
            return 10
        'Remove a list of pages in the GUI.\n\n        Args:\n            page_names (list): List of page names (str) to display, such as\n                               ["Weather.qml", "Forecast.qml", "Other.qml"]\n        '
        if not isinstance(page_names, list):
            raise ValueError('page_names must be a list')
        page_urls = []
        for name in page_names:
            if name.startswith('SYSTEM'):
                page = resolve_resource_file(join('ui', name))
            else:
                page = self.skill.find_resource(name, 'ui')
            if page:
                if self.config.get('remote'):
                    page_urls.append(self.remote_url + '/' + page)
                else:
                    page_urls.append('file://' + page)
            else:
                raise FileNotFoundError('Unable to find page: {}'.format(name))
        self.skill.bus.emit(Message('gui.page.delete', {'page': page_urls, '__from': self.skill.skill_id}))

    def show_text(self, text, title=None, override_idle=None, override_animations=False):
        if False:
            for i in range(10):
                print('nop')
        "Display a GUI page for viewing simple text.\n\n        Args:\n            text (str): Main text content.  It will auto-paginate\n            title (str): A title to display above the text content.\n            override_idle (boolean, int):\n                True: Takes over the resting page indefinitely\n                (int): Delays resting page for the specified number of\n                       seconds.\n            override_animations (boolean):\n                True: Disables showing all platform skill animations.\n                False: 'Default' always show animations.\n        "
        self['text'] = text
        self['title'] = title
        self.show_page('SYSTEM_TextFrame.qml', override_idle, override_animations)

    def show_image(self, url, caption=None, title=None, fill=None, override_idle=None, override_animations=False):
        if False:
            for i in range(10):
                print('nop')
        "Display a GUI page for viewing an image.\n\n        Args:\n            url (str): Pointer to the image\n            caption (str): A caption to show under the image\n            title (str): A title to display above the image content\n            fill (str): Fill type supports 'PreserveAspectFit',\n            'PreserveAspectCrop', 'Stretch'\n            override_idle (boolean, int):\n                True: Takes over the resting page indefinitely\n                (int): Delays resting page for the specified number of\n                       seconds.\n            override_animations (boolean):\n                True: Disables showing all platform skill animations.\n                False: 'Default' always show animations.\n        "
        self['image'] = url
        self['title'] = title
        self['caption'] = caption
        self['fill'] = fill
        self.show_page('SYSTEM_ImageFrame.qml', override_idle, override_animations)

    def show_animated_image(self, url, caption=None, title=None, fill=None, override_idle=None, override_animations=False):
        if False:
            i = 10
            return i + 15
        "Display a GUI page for viewing an image.\n\n        Args:\n            url (str): Pointer to the .gif image\n            caption (str): A caption to show under the image\n            title (str): A title to display above the image content\n            fill (str): Fill type supports 'PreserveAspectFit',\n            'PreserveAspectCrop', 'Stretch'\n            override_idle (boolean, int):\n                True: Takes over the resting page indefinitely\n                (int): Delays resting page for the specified number of\n                       seconds.\n            override_animations (boolean):\n                True: Disables showing all platform skill animations.\n                False: 'Default' always show animations.\n        "
        self['image'] = url
        self['title'] = title
        self['caption'] = caption
        self['fill'] = fill
        self.show_page('SYSTEM_AnimatedImageFrame.qml', override_idle, override_animations)

    def show_html(self, html, resource_url=None, override_idle=None, override_animations=False):
        if False:
            i = 10
            return i + 15
        "Display an HTML page in the GUI.\n\n        Args:\n            html (str): HTML text to display\n            resource_url (str): Pointer to HTML resources\n            override_idle (boolean, int):\n                True: Takes over the resting page indefinitely\n                (int): Delays resting page for the specified number of\n                       seconds.\n            override_animations (boolean):\n                True: Disables showing all platform skill animations.\n                False: 'Default' always show animations.\n        "
        self['html'] = html
        self['resourceLocation'] = resource_url
        self.show_page('SYSTEM_HtmlFrame.qml', override_idle, override_animations)

    def show_url(self, url, override_idle=None, override_animations=False):
        if False:
            print('Hello World!')
        "Display an HTML page in the GUI.\n\n        Args:\n            url (str): URL to render\n            override_idle (boolean, int):\n                True: Takes over the resting page indefinitely\n                (int): Delays resting page for the specified number of\n                       seconds.\n            override_animations (boolean):\n                True: Disables showing all platform skill animations.\n                False: 'Default' always show animations.\n        "
        self['url'] = url
        self.show_page('SYSTEM_UrlFrame.qml', override_idle, override_animations)

    def release(self):
        if False:
            print('Hello World!')
        'Signal that this skill is no longer using the GUI,\n        allow different platforms to properly handle this event.\n        Also calls self.clear() to reset the state variables\n        Platforms can close the window or go back to previous page'
        self.clear()
        self.skill.bus.emit(Message('mycroft.gui.screen.close', {'skill_id': self.skill.skill_id}))

    def shutdown(self):
        if False:
            while True:
                i = 10
        'Shutdown gui interface.\n\n        Clear pages loaded through this interface and remove the skill\n        reference to make ref counting warning more precise.\n        '
        self.release()
        self.skill = None