from gi import pygtkcompat
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
GObject.threads_init()
Gst.init(None)
gst = Gst
print('Using pygtkcompat and Gst from gi')
pygtkcompat.enable()
pygtkcompat.enable_gtk(version='3.0')
import gtk

class DemoApp(object):
    """GStreamer/PocketSphinx Demo Application"""

    def __init__(self):
        if False:
            return 10
        'Initialize a DemoApp object'
        self.init_gui()
        self.init_gst()

    def init_gui(self):
        if False:
            while True:
                i = 10
        'Initialize the GUI components'
        self.window = gtk.Window()
        self.window.connect('delete-event', gtk.main_quit)
        self.window.set_default_size(400, 200)
        self.window.set_border_width(10)
        vbox = gtk.VBox()
        self.textbuf = gtk.TextBuffer()
        self.text = gtk.TextView(buffer=self.textbuf)
        self.text.set_wrap_mode(gtk.WRAP_WORD)
        vbox.pack_start(self.text)
        self.button = gtk.ToggleButton('Speak')
        self.button.connect('clicked', self.button_clicked)
        vbox.pack_start(self.button, False, False, 5)
        self.window.add(vbox)
        self.window.show_all()

    def init_gst(self):
        if False:
            while True:
                i = 10
        'Initialize the speech components'
        self.pipeline = gst.parse_launch('autoaudiosrc ! audioconvert ! audioresample ! pocketsphinx ! fakesink')
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message::element', self.element_message)
        self.pipeline.set_state(gst.State.PAUSED)

    def element_message(self, bus, msg):
        if False:
            print('Hello World!')
        'Receive element messages from the bus.'
        msgtype = msg.get_structure().get_name()
        if msgtype != 'pocketsphinx':
            return
        if msg.get_structure().get_value('final'):
            self.final_result(msg.get_structure().get_value('hypothesis'), msg.get_structure().get_value('confidence'))
            self.pipeline.set_state(gst.State.PAUSED)
            self.button.set_active(False)
        elif msg.get_structure().get_value('hypothesis'):
            self.partial_result(msg.get_structure().get_value('hypothesis'))

    def partial_result(self, hyp):
        if False:
            for i in range(10):
                print('nop')
        'Delete any previous selection, insert text and select it.'
        self.textbuf.begin_user_action()
        self.textbuf.delete_selection(True, self.text.get_editable())
        self.textbuf.insert_at_cursor(hyp)
        ins = self.textbuf.get_insert()
        iter = self.textbuf.get_iter_at_mark(ins)
        iter.backward_chars(len(hyp))
        self.textbuf.move_mark(ins, iter)
        self.textbuf.end_user_action()

    def final_result(self, hyp, confidence):
        if False:
            i = 10
            return i + 15
        'Insert the final result.'
        self.textbuf.begin_user_action()
        self.textbuf.delete_selection(True, self.text.get_editable())
        self.textbuf.insert_at_cursor(hyp)
        self.textbuf.end_user_action()

    def button_clicked(self, button):
        if False:
            while True:
                i = 10
        'Handle button presses.'
        if button.get_active():
            button.set_label('Stop')
            self.pipeline.set_state(gst.State.PLAYING)
        else:
            button.set_label('Speak')
            self.pipeline.set_state(gst.State.PAUSED)
app = DemoApp()
gtk.main()