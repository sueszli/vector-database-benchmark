from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextBlockFormat, QTextCursor
from picard.config import BoolOption, IntOption, TextOption, get_config
from picard.track import TagGenreFilter
from picard.ui.options import OptionsPage, register_options_page
from picard.ui.ui_options_genres import Ui_GenresOptionsPage
TOOLTIP_GENRES_FILTER = N_('<html><head/><body>\n<p>Lines not starting with <b>-</b> or <b>+</b> are ignored.</p>\n<p>One expression per line, case-insensitive</p>\n<p>Examples:</p>\n<p><b>\n#comment<br/>\n!comment<br/>\ncomment\n</b></p>\n<p><u>Strict filtering:</u></p>\n<p>\n<b>-word</b>: exclude <i>word</i><br/>\n<b>+word</b>: include <i>word</i>\n</p>\n<p><u>Wildcard filtering:</u></p>\n<p>\n<b>-*word</b>: exclude all genres ending with <i>word</i><br/>\n<b>+word*</b>: include all genres starting with <i>word</i><br/>\n<b>+wor?</b>: include all genres starting with <i>wor</i> and ending with an arbitrary character<br/>\n<b>+wor[dk]</b>: include all genres starting with <i>wor</i> and ending with <i>d</i> or <i>k</i><br/>\n<b>-w*rd</b>: exclude all genres starting with <i>w</i> and ending with <i>rd</i>\n</p>\n<p><u>Regular expressions filtering (Python re syntax):</u></p>\n<p><b>-/^w.rd+/</b>: exclude genres starting with <i>w</i> followed by any character, then <i>r</i> followed by at least one <i>d</i>\n</p>\n</body></html>')
TOOLTIP_TEST_GENRES_FILTER = N_('<html><head/><body>\n<p>You can add genres to test filters against, one per line.<br/>\nThis playground will not be preserved on Options exit.\n</p>\n<p>\nRed background means the tag will be skipped.<br/>\nGreen background means the tag will be kept.\n</p>\n</body></html>')

class GenresOptionsPage(OptionsPage):
    NAME = 'genres'
    TITLE = N_('Genres')
    PARENT = 'metadata'
    SORT_ORDER = 20
    ACTIVE = True
    HELP_URL = '/config/options_genres.html'
    options = [BoolOption('setting', 'use_genres', False), IntOption('setting', 'max_genres', 5), IntOption('setting', 'min_genre_usage', 90), TextOption('setting', 'genres_filter', '-seen live\n-favorites\n-fixme\n-owned'), TextOption('setting', 'join_genres', ''), BoolOption('setting', 'only_my_genres', False), BoolOption('setting', 'artists_genres', False), BoolOption('setting', 'folksonomy_tags', False)]

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.ui = Ui_GenresOptionsPage()
        self.ui.setupUi(self)
        self.ui.genres_filter.setToolTip(_(TOOLTIP_GENRES_FILTER))
        self.ui.genres_filter.textChanged.connect(self.update_test_genres_filter)
        self.ui.test_genres_filter.setToolTip(_(TOOLTIP_TEST_GENRES_FILTER))
        self.ui.test_genres_filter.textChanged.connect(self.update_test_genres_filter)
        self.fmt_keep = QTextBlockFormat()
        self.fmt_keep.setBackground(Qt.GlobalColor.green)
        self.fmt_skip = QTextBlockFormat()
        self.fmt_skip.setBackground(Qt.GlobalColor.red)
        self.fmt_clear = QTextBlockFormat()
        self.fmt_clear.clearBackground()

    def load(self):
        if False:
            while True:
                i = 10
        config = get_config()
        self.ui.use_genres.setChecked(config.setting['use_genres'])
        self.ui.max_genres.setValue(config.setting['max_genres'])
        self.ui.min_genre_usage.setValue(config.setting['min_genre_usage'])
        self.ui.join_genres.setEditText(config.setting['join_genres'])
        self.ui.genres_filter.setPlainText(config.setting['genres_filter'])
        self.ui.only_my_genres.setChecked(config.setting['only_my_genres'])
        self.ui.artists_genres.setChecked(config.setting['artists_genres'])
        self.ui.folksonomy_tags.setChecked(config.setting['folksonomy_tags'])

    def save(self):
        if False:
            while True:
                i = 10
        config = get_config()
        config.setting['use_genres'] = self.ui.use_genres.isChecked()
        config.setting['max_genres'] = self.ui.max_genres.value()
        config.setting['min_genre_usage'] = self.ui.min_genre_usage.value()
        config.setting['join_genres'] = self.ui.join_genres.currentText()
        config.setting['genres_filter'] = self.ui.genres_filter.toPlainText()
        config.setting['only_my_genres'] = self.ui.only_my_genres.isChecked()
        config.setting['artists_genres'] = self.ui.artists_genres.isChecked()
        config.setting['folksonomy_tags'] = self.ui.folksonomy_tags.isChecked()

    def update_test_genres_filter(self):
        if False:
            i = 10
            return i + 15
        test_text = self.ui.test_genres_filter.toPlainText()
        filters = self.ui.genres_filter.toPlainText()
        tagfilter = TagGenreFilter(filters)
        self.ui.label_test_genres_filter_error.setText('\n'.join(tagfilter.format_errors()))

        def set_line_fmt(lineno, textformat):
            if False:
                while True:
                    i = 10
            obj = self.ui.test_genres_filter
            if lineno < 0:
                cursor = obj.textCursor()
            else:
                cursor = QTextCursor(obj.document().findBlockByNumber(lineno))
            obj.blockSignals(True)
            cursor.setBlockFormat(textformat)
            obj.blockSignals(False)
        set_line_fmt(-1, self.fmt_clear)
        for (lineno, line) in enumerate(test_text.splitlines()):
            line = line.strip()
            fmt = self.fmt_clear
            if line:
                if tagfilter.skip(line):
                    fmt = self.fmt_skip
                else:
                    fmt = self.fmt_keep
            set_line_fmt(lineno, fmt)
register_options_page(GenresOptionsPage)