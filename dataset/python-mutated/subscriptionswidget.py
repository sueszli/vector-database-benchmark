from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel, QWidget
from tribler.gui.sentry_mixin import AddBreadcrumbOnShowMixin
from tribler.gui.utilities import connect, format_votes_rich_text, get_votes_rating_description, tr
from tribler.gui.widgets.tablecontentdelegate import DARWIN, WINDOWS

class SubscriptionsWidget(AddBreadcrumbOnShowMixin, QWidget):
    """
    This widget shows a favorite button and the number of subscriptions that a specific channel has.
    """

    def __init__(self, parent):
        if False:
            i = 10
            return i + 15
        QWidget.__init__(self, parent)
        self.subscribe_button = None
        self.initialized = False
        self.contents_widget = None
        self.channel_rating_label = None

    def initialize(self, contents_widget):
        if False:
            print('Hello World!')
        if not self.initialized:
            self.contents_widget = contents_widget
            self.subscribe_button = self.findChild(QWidget, 'subscribe_button')
            self.channel_rating_label = self.findChild(QLabel, 'channel_rating_label')
            self.channel_rating_label.setTextFormat(Qt.RichText)
            connect(self.subscribe_button.clicked, self.on_subscribe_button_click)
            self.subscribe_button.setToolTip(tr('Click to subscribe/unsubscribe'))
            connect(self.subscribe_button.toggled, self._adjust_tooltip)
            self.initialized = True

    def _adjust_tooltip(self, toggled):
        if False:
            return 10
        tooltip = (tr('Subscribed.') if toggled else tr('Not subscribed.')) + tr('\n(Click to unsubscribe)')
        self.subscribe_button.setToolTip(tooltip)

    def update_subscribe_button_if_channel_matches(self, changed_channels_list):
        if False:
            while True:
                i = 10
        if not (self.contents_widget.model and self.contents_widget.model.channel_info.get('public_key')):
            return
        for channel_info in changed_channels_list:
            if self.contents_widget.model.channel_info['public_key'] == channel_info['public_key'] and self.contents_widget.model.channel_info['id'] == channel_info['id']:
                self.update_subscribe_button(remote_response=channel_info)
                return

    def update_subscribe_button(self, remote_response=None):
        if False:
            i = 10
            return i + 15
        if self.isHidden():
            return
        if remote_response and 'subscribed' in remote_response:
            self.contents_widget.model.channel_info['subscribed'] = remote_response['subscribed']
        self.subscribe_button.setChecked(bool(remote_response['subscribed']))
        self._adjust_tooltip(bool(remote_response['subscribed']))
        votes = remote_response['votes']
        self.channel_rating_label.setText(format_votes_rich_text(votes))
        if DARWIN or WINDOWS:
            font = QFont()
            font.setLetterSpacing(QFont.PercentageSpacing, 60.0)
            self.channel_rating_label.setFont(font)
        self.channel_rating_label.setToolTip(get_votes_rating_description(votes))

    def on_subscribe_button_click(self, checked):
        if False:
            for i in range(10):
                print('nop')
        self.subscribe_button.setCheckedInstant(bool(self.contents_widget.model.channel_info['subscribed']))
        channel_info = self.contents_widget.model.channel_info
        if channel_info['subscribed']:
            self.window().on_channel_unsubscribe(channel_info)
        else:
            self.window().on_channel_subscribe(channel_info)