__license__ = 'GPL v3'
__copyright__ = '2014, Kovid Goyal <kovid at kovidgoyal.net>'
import re
import textwrap
from bisect import bisect
from functools import partial
from qt.core import QAbstractItemModel, QAbstractItemView, QAbstractListModel, QApplication, QCheckBox, QDialogButtonBox, QGridLayout, QHBoxLayout, QIcon, QInputMethodEvent, QLabel, QListView, QMenu, QMimeData, QModelIndex, QPen, QPushButton, QSize, QSizePolicy, QSplitter, QStyledItemDelegate, Qt, QToolButton, QTreeView, pyqtSignal
from calibre.gui2.tweak_book import tprefs
from calibre.gui2.tweak_book.widgets import Dialog
from calibre.gui2.widgets import BusyCursor
from calibre.gui2.widgets2 import HistoryLineEdit2
from calibre.startup import connect_lambda
from calibre.utils.icu import safe_chr as codepoint_to_chr
from calibre.utils.unicode_names import character_name_from_code, points_for_word
from calibre_extensions.progress_indicator import set_no_activate_on_click
ROOT = QModelIndex()
non_printing = {160: 'nbsp', 8192: 'nqsp', 8193: 'mqsp', 8194: 'ensp', 8195: 'emsp', 8196: '3/msp', 8197: '4/msp', 8198: '6/msp', 8199: 'fsp', 8200: 'psp', 8201: 'thsp', 8202: 'hsp', 8203: 'zwsp', 8204: 'zwnj', 8205: 'zwj', 8206: 'lrm', 8207: 'rlm', 8232: 'lsep', 8233: 'psep', 8234: 'rle', 8235: 'lre', 8236: 'pdp', 8237: 'lro', 8238: 'rlo', 8239: 'nnbsp', 8287: 'mmsp', 8288: 'wj', 8289: 'fa', 8290: 'x', 8291: ',', 8292: '+', 8298: 'iss', 8299: 'ass', 8300: 'iafs', 8301: 'aafs', 8302: 'nads', 8303: 'nods', 32: 'sp', 127: 'del', 11834: '2m', 11835: '3m', 173: 'shy'}

def search_for_chars(query, and_tokens=False):
    if False:
        i = 10
        return i + 15
    ans = set()
    for (i, token) in enumerate(query.split()):
        token = token.lower()
        m = re.match('(?:[u]\\+)([a-f0-9]+)', token)
        if m is not None:
            chars = {int(m.group(1), 16)}
        else:
            chars = points_for_word(token)
        if chars is not None:
            if and_tokens:
                ans = chars if i == 0 else ans & chars
            else:
                ans |= chars
    return sorted(ans)

class CategoryModel(QAbstractItemModel):

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        QAbstractItemModel.__init__(self, parent)
        self.categories = ((_('Favorites'), ()), (_('European scripts'), ((_('Armenian'), (1328, 1423)), (_('Armenian ligatures'), (64275, 64279)), (_('Coptic'), (11392, 11519)), (_('Coptic in Greek block'), (994, 1007)), (_('Cypriot syllabary'), (67584, 67647)), (_('Cyrillic'), (1024, 1279)), (_('Cyrillic supplement'), (1280, 1327)), (_('Cyrillic extended A'), (11744, 11775)), (_('Cyrillic extended B'), (42560, 42655)), (_('Georgian'), (4256, 4351)), (_('Georgian supplement'), (11520, 11567)), (_('Glagolitic'), (11264, 11359)), (_('Gothic'), (66352, 66383)), (_('Greek and Coptic'), (880, 1023)), (_('Greek extended'), (7936, 8191)), (_('Latin, Basic & Latin-1 supplement'), (32, 255)), (_('Latin extended A'), (256, 383)), (_('Latin extended B'), (384, 591)), (_('Latin extended C'), (11360, 11391)), (_('Latin extended D'), (42784, 43007)), (_('Latin extended additional'), (7680, 7935)), (_('Latin ligatures'), (64256, 64262)), (_('Fullwidth Latin letters'), (65280, 65374)), (_('Linear B syllabary'), (65536, 65663)), (_('Linear B ideograms'), (65664, 65791)), (_('Ogham'), (5760, 5791)), (_('Old italic'), (66304, 66351)), (_('Phaistos disc'), (66000, 66047)), (_('Runic'), (5792, 5887)), (_('Shavian'), (66640, 66687)))), (_('Phonetic symbols'), ((_('IPA extensions'), (592, 687)), (_('Phonetic extensions'), (7424, 7551)), (_('Phonetic extensions supplement'), (7552, 7615)), (_('Modifier tone letters'), (42752, 42783)), (_('Spacing modifier letters'), (688, 767)), (_('Superscripts and subscripts'), (8304, 8351)))), (_('Combining diacritics'), ((_('Combining diacritical marks'), (768, 879)), (_('Combining diacritical marks for symbols'), (8400, 8447)), (_('Combining diacritical marks supplement'), (7616, 7679)), (_('Combining half marks'), (65056, 65071)))), (_('African scripts'), ((_('Bamum'), (42656, 42751)), (_('Bamum supplement'), (92160, 92735)), (_('Egyptian hieroglyphs'), (77824, 78895)), (_('Ethiopic'), (4608, 4991)), (_('Ethiopic supplement'), (4992, 5023)), (_('Ethiopic extended'), (11648, 11743)), (_('Ethiopic extended A'), (43776, 43823)), (_('Meroitic cursive'), (68000, 68095)), (_('Meroitic hieroglyphs'), (67968, 67999)), (_("N'Ko"), (1984, 2047)), (_('Osmanya'), (66688, 66735)), (_('Tifinagh'), (11568, 11647)), (_('Vai'), (42240, 42559)))), (_('Middle Eastern scripts'), ((_('Arabic'), (1536, 1791)), (_('Arabic supplement'), (1872, 1919)), (_('Arabic extended A'), (2208, 2303)), (_('Arabic presentation forms A'), (64336, 65023)), (_('Arabic presentation forms B'), (65136, 65279)), (_('Avestan'), (68352, 68415)), (_('Carian'), (66208, 66271)), (_('Cuneiform'), (73728, 74751)), (_('Cuneiform numbers and punctuation'), (74752, 74879)), (_('Hebrew'), (1424, 1535)), (_('Hebrew presentation forms'), (64285, 64335)), (_('Imperial Aramaic'), (67648, 67679)), (_('Inscriptional Pahlavi'), (68448, 68479)), (_('Inscriptional Parthian'), (68416, 68447)), (_('Lycian'), (66176, 66207)), (_('Lydian'), (67872, 67903)), (_('Mandaic'), (2112, 2143)), (_('Old Persian'), (66464, 66527)), (_('Old South Arabian'), (68192, 68223)), (_('Phoenician'), (67840, 67871)), (_('Samaritan'), (2048, 2111)), (_('Syriac'), (1792, 1871)), (_('Ugaritic'), (66432, 66463)))), (_('Central Asian scripts'), ((_('Mongolian'), (6144, 6319)), (_('Old Turkic'), (68608, 68687)), (_('Phags-pa'), (43072, 43135)), (_('Tibetan'), (3840, 4095)))), (_('South Asian scripts'), ((_('Bengali'), (2432, 2559)), (_('Brahmi'), (69632, 69759)), (_('Chakma'), (69888, 69967)), (_('Devanagari'), (2304, 2431)), (_('Devanagari extended'), (43232, 43263)), (_('Gujarati'), (2688, 2815)), (_('Gurmukhi'), (2560, 2687)), (_('Kaithi'), (69760, 69839)), (_('Kannada'), (3200, 3327)), (_('Kharoshthi'), (68096, 68191)), (_('Lepcha'), (7168, 7247)), (_('Limbu'), (6400, 6479)), (_('Malayalam'), (3328, 3455)), (_('Meetei Mayek'), (43968, 44031)), (_('Meetei Mayek extensions'), (43744, 43759)), (_('Ol Chiki'), (7248, 7295)), (_('Oriya'), (2816, 2943)), (_('Saurashtra'), (43136, 43231)), (_('Sinhala'), (3456, 3583)), (_('Sharada'), (70016, 70111)), (_('Sora Sompeng'), (69840, 69887)), (_('Syloti Nagri'), (43008, 43055)), (_('Takri'), (71296, 71375)), (_('Tamil'), (2944, 3071)), (_('Telugu'), (3072, 3199)), (_('Thaana'), (1920, 1983)), (_('Vedic extensions'), (7376, 7423)))), (_('Southeast Asian scripts'), ((_('Balinese'), (6912, 7039)), (_('Batak'), (7104, 7167)), (_('Buginese'), (6656, 6687)), (_('Cham'), (43520, 43615)), (_('Javanese'), (43392, 43487)), (_('Kayah Li'), (43264, 43311)), (_('Khmer'), (6016, 6143)), (_('Khmer symbols'), (6624, 6655)), (_('Lao'), (3712, 3839)), (_('Myanmar'), (4096, 4255)), (_('Myanmar extended A'), (43616, 43647)), (_('New Tai Lue'), (6528, 6623)), (_('Rejang'), (43312, 43359)), (_('Sundanese'), (7040, 7103)), (_('Sundanese supplement'), (7360, 7375)), (_('Tai Le'), (6480, 6527)), (_('Tai Tham'), (6688, 6831)), (_('Tai Viet'), (43648, 43743)), (_('Thai'), (3584, 3711)))), (_('Philippine scripts'), ((_('Buhid'), (5952, 5983)), (_('Hanunoo'), (5920, 5951)), (_('Tagalog'), (5888, 5919)), (_('Tagbanwa'), (5984, 6015)))), (_('East Asian scripts'), ((_('Bopomofo'), (12544, 12591)), (_('Bopomofo extended'), (12704, 12735)), (_('CJK Unified ideographs'), (19968, 40959)), (_('CJK Unified ideographs extension A'), (13312, 19903)), (_('CJK Unified ideographs extension B'), (131072, 173791)), (_('CJK Unified ideographs extension C'), (173824, 177983)), (_('CJK Unified ideographs extension D'), (177984, 178207)), (_('CJK compatibility ideographs'), (63744, 64255)), (_('CJK compatibility ideographs supplement'), (194560, 195103)), (_('Kangxi radicals'), (12032, 12255)), (_('CJK radicals supplement'), (11904, 12031)), (_('CJK strokes'), (12736, 12783)), (_('Ideographic description characters'), (12272, 12287)), (_('Hiragana'), (12352, 12447)), (_('Katakana'), (12448, 12543)), (_('Katakana phonetic extensions'), (12784, 12799)), (_('Kana supplement'), (110592, 110847)), (_('Halfwidth Katakana'), (65381, 65439)), (_('Kanbun'), (12688, 12703)), (_('Hangul syllables'), (44032, 55215)), (_('Hangul Jamo'), (4352, 4607)), (_('Hangul Jamo extended A'), (43360, 43391)), (_('Hangul Jamo extended B'), (55216, 55295)), (_('Hangul compatibility Jamo'), (12592, 12687)), (_('Halfwidth Jamo'), (65440, 65500)), (_('Lisu'), (42192, 42239)), (_('Miao'), (93952, 94111)), (_('Yi syllables'), (40960, 42127)), (_('Yi radicals'), (42128, 42191)))), (_('American scripts'), ((_('Cherokee'), (5024, 5119)), (_('Deseret'), (66560, 66639)), (_('Unified Canadian aboriginal syllabics'), (5120, 5759)), (_('UCAS extended'), (6320, 6399)))), (_('Other'), ((_('Alphabetic presentation forms'), (64256, 64335)), (_('Halfwidth and Fullwidth forms'), (65280, 65519)))), (_('Punctuation'), ((_('General punctuation'), (8192, 8303)), (_('ASCII punctuation'), (33, 127)), (_('Cuneiform numbers and punctuation'), (74752, 74879)), (_('Latin-1 punctuation'), (161, 191)), (_('Small form variants'), (65104, 65135)), (_('Supplemental punctuation'), (11776, 11903)), (_('CJK symbols and punctuation'), (12288, 12351)), (_('CJK compatibility forms'), (65072, 65103)), (_('Fullwidth ASCII punctuation'), (65281, 65376)), (_('Vertical forms'), (65040, 65055)))), (_('Alphanumeric symbols'), ((_('Arabic mathematical alphabetic symbols'), (126464, 126719)), (_('Letterlike symbols'), (8448, 8527)), (_('Roman symbols'), (65936, 65999)), (_('Mathematical alphanumeric symbols'), (119808, 120831)), (_('Enclosed alphanumerics'), (9312, 9471)), (_('Enclosed alphanumeric supplement'), (127232, 127487)), (_('Enclosed CJK letters and months'), (12800, 13055)), (_('Enclosed ideographic supplement'), (127488, 127743)), (_('CJK compatibility'), (13056, 13311)))), (_('Technical symbols'), ((_('Miscellaneous technical'), (8960, 9215)), (_('Control pictures'), (9216, 9279)), (_('Optical character recognition'), (9280, 9311)))), (_('Numbers and digits'), ((_('Aegean numbers'), (65792, 65855)), (_('Ancient Greek numbers'), (65856, 65935)), (_('Common Indic number forms'), (43056, 43071)), (_('Counting rod numerals'), (119648, 119679)), (_('Cuneiform numbers and punctuation'), (74752, 74879)), (_('Fullwidth ASCII digits'), (65296, 65305)), (_('Number forms'), (8528, 8591)), (_('Rumi numeral symbols'), (69216, 69247)), (_('Superscripts and subscripts'), (8304, 8351)))), (_('Mathematical symbols'), ((_('Arrows'), (8592, 8703)), (_('Supplemental arrows A'), (10224, 10239)), (_('Supplemental arrows B'), (10496, 10623)), (_('Miscellaneous symbols and arrows'), (11008, 11263)), (_('Mathematical alphanumeric symbols'), (119808, 120831)), (_('Letterlike symbols'), (8448, 8527)), (_('Mathematical operators'), (8704, 8959)), (_('Miscellaneous mathematical symbols A'), (10176, 10223)), (_('Miscellaneous mathematical symbols B'), (10624, 10751)), (_('Supplemental mathematical operators'), (10752, 11007)), (_('Ceilings and floors'), (8968, 8971)), (_('Geometric shapes'), (9632, 9727)), (_('Box drawing'), (9472, 9599)), (_('Block elements'), (9600, 9631)))), (_('Musical symbols'), ((_('Musical symbols'), (119040, 119295)), (_('More musical symbols'), (9833, 9839)), (_('Ancient Greek musical notation'), (119296, 119375)), (_('Byzantine musical symbols'), (118784, 119039)))), (_('Game symbols'), ((_('Chess'), (9812, 9823)), (_('Domino tiles'), (127024, 127135)), (_('Draughts'), (9920, 9923)), (_('Japanese chess'), (9750, 9751)), (_('Mahjong tiles'), (126976, 127023)), (_('Playing cards'), (127136, 127231)), (_('Playing card suits'), (9824, 9831)))), (_('Other symbols'), ((_('Alchemical symbols'), (128768, 128895)), (_('Ancient symbols'), (65936, 65999)), (_('Braille patterns'), (10240, 10495)), (_('Currency symbols'), (8352, 8399)), (_('Combining diacritical marks for symbols'), (8400, 8447)), (_('Dingbats'), (9984, 10175)), (_('Emoticons'), (128512, 128591)), (_('Miscellaneous symbols'), (9728, 9983)), (_('Miscellaneous symbols and arrows'), (11008, 11263)), (_('Miscellaneous symbols and pictographs'), (127744, 128511)), (_('Yijing hexagram symbols'), (19904, 19967)), (_('Yijing mono and digrams'), (9866, 9871)), (_('Yijing trigrams'), (9776, 9783)), (_('Tai Xuan Jing symbols'), (119552, 119647)), (_('Transport and map symbols'), (128640, 128767)))), (_('Other'), ((_('Specials'), (65520, 65535)), (_('Tags'), (917504, 917631)), (_('Variation selectors'), (65024, 65039)), (_('Variation selectors supplement'), (917760, 917999)))))
        self.category_map = {}
        self.starts = []
        for (tlname, items) in self.categories[1:]:
            for (name, (start, end)) in items:
                self.category_map[start] = (tlname, name)
                self.starts.append(start)
        self.starts.sort()
        self.bold_font = f = QApplication.font()
        f.setBold(True)
        self.fav_icon = QIcon.ic('rating.png')

    def columnCount(self, parent=ROOT):
        if False:
            return 10
        return 1

    def rowCount(self, parent=ROOT):
        if False:
            while True:
                i = 10
        if not parent.isValid():
            return len(self.categories)
        r = parent.row()
        pid = parent.internalId()
        if pid == 0 and -1 < r < len(self.categories):
            return len(self.categories[r][1])
        return 0

    def index(self, row, column, parent=ROOT):
        if False:
            return 10
        if not parent.isValid():
            return self.createIndex(row, column) if -1 < row < len(self.categories) else ROOT
        try:
            return self.createIndex(row, column, parent.row() + 1) if -1 < row < len(self.categories[parent.row()][1]) else ROOT
        except IndexError:
            return ROOT

    def parent(self, index):
        if False:
            for i in range(10):
                print('nop')
        if not index.isValid():
            return ROOT
        pid = index.internalId()
        if pid == 0:
            return ROOT
        return self.index(pid - 1, 0)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if False:
            for i in range(10):
                print('nop')
        if not index.isValid():
            return None
        pid = index.internalId()
        if pid == 0:
            if role == Qt.ItemDataRole.DisplayRole:
                return self.categories[index.row()][0]
            if role == Qt.ItemDataRole.FontRole:
                return self.bold_font
            if role == Qt.ItemDataRole.DecorationRole and index.row() == 0:
                return self.fav_icon
        elif role == Qt.ItemDataRole.DisplayRole:
            item = self.categories[pid - 1][1][index.row()]
            return item[0]
        return None

    def get_range(self, index):
        if False:
            for i in range(10):
                print('nop')
        if index.isValid():
            pid = index.internalId()
            if pid == 0:
                if index.row() == 0:
                    return (_('Favorites'), list(tprefs['charmap_favorites']))
            else:
                item = self.categories[pid - 1][1][index.row()]
                return (item[0], list(range(item[1][0], item[1][1] + 1)))

    def get_char_info(self, char_code):
        if False:
            i = 10
            return i + 15
        ipos = bisect(self.starts, char_code) - 1
        try:
            (category, subcategory) = self.category_map[self.starts[ipos]]
        except IndexError:
            category = subcategory = _('Unknown')
        return (category, subcategory, character_name_from_code(char_code))

class CategoryDelegate(QStyledItemDelegate):

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        QStyledItemDelegate.__init__(self, parent)

    def sizeHint(self, option, index):
        if False:
            i = 10
            return i + 15
        ans = QStyledItemDelegate.sizeHint(self, option, index)
        if not index.parent().isValid():
            ans += QSize(0, 6)
        return ans

class CategoryView(QTreeView):
    category_selected = pyqtSignal(object, object)

    def __init__(self, parent=None):
        if False:
            return 10
        QTreeView.__init__(self, parent)
        self.setHeaderHidden(True)
        self.setAnimated(True)
        self.activated.connect(self.item_activated)
        self.clicked.connect(self.item_activated)
        set_no_activate_on_click(self)
        self.initialized = False
        self.setExpandsOnDoubleClick(False)

    def item_activated(self, index):
        if False:
            while True:
                i = 10
        ans = self._model.get_range(index)
        if ans is not None:
            self.category_selected.emit(*ans)
        elif self.isExpanded(index):
            self.collapse(index)
        else:
            self.expand(index)

    def get_chars(self):
        if False:
            while True:
                i = 10
        ans = self._model.get_range(self.currentIndex())
        if ans is not None:
            self.category_selected.emit(*ans)

    def initialize(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.initialized:
            self._model = m = CategoryModel(self)
            self.setModel(m)
            self.setCurrentIndex(m.index(0, 0))
            self.item_activated(m.index(0, 0))
            self._delegate = CategoryDelegate(self)
            self.setItemDelegate(self._delegate)
            self.initialized = True

class CharModel(QAbstractListModel):

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        QAbstractListModel.__init__(self, parent)
        self.chars = []
        self.allow_dnd = False

    def rowCount(self, parent=ROOT):
        if False:
            return 10
        return len(self.chars)

    def data(self, index, role):
        if False:
            print('Hello World!')
        if role == Qt.ItemDataRole.UserRole and -1 < index.row() < len(self.chars):
            return self.chars[index.row()]
        return None

    def flags(self, index):
        if False:
            i = 10
            return i + 15
        ans = Qt.ItemFlag.ItemIsEnabled
        if self.allow_dnd:
            ans |= Qt.ItemFlag.ItemIsSelectable
            ans |= Qt.ItemFlag.ItemIsDragEnabled if index.isValid() else Qt.ItemFlag.ItemIsDropEnabled
        return ans

    def supportedDropActions(self):
        if False:
            i = 10
            return i + 15
        return Qt.DropAction.MoveAction

    def mimeTypes(self):
        if False:
            i = 10
            return i + 15
        return ['application/calibre_charcode_indices']

    def mimeData(self, indexes):
        if False:
            for i in range(10):
                print('nop')
        data = ','.join((str(i.row()) for i in indexes))
        md = QMimeData()
        md.setData('application/calibre_charcode_indices', data.encode('utf-8'))
        return md

    def dropMimeData(self, md, action, row, column, parent):
        if False:
            for i in range(10):
                print('nop')
        if action != Qt.DropAction.MoveAction or not md.hasFormat('application/calibre_charcode_indices') or row < 0 or (column != 0):
            return False
        indices = list(map(int, bytes(md.data('application/calibre_charcode_indices')).decode('ascii').split(',')))
        codes = [self.chars[x] for x in indices]
        for x in indices:
            self.chars[x] = None
        for x in reversed(codes):
            self.chars.insert(row, x)
        self.beginResetModel()
        self.chars = [x for x in self.chars if x is not None]
        self.endResetModel()
        tprefs['charmap_favorites'] = list(self.chars)
        return True

class CharDelegate(QStyledItemDelegate):

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        QStyledItemDelegate.__init__(self, parent)
        self.item_size = QSize(32, 32)
        self.np_pat = re.compile('(sp|j|nj|ss|fs|ds)$')

    def sizeHint(self, option, index):
        if False:
            print('Hello World!')
        return self.item_size

    def paint(self, painter, option, index):
        if False:
            return 10
        QStyledItemDelegate.paint(self, painter, option, index)
        try:
            charcode = int(index.data(Qt.ItemDataRole.UserRole))
        except (TypeError, ValueError):
            return
        painter.save()
        try:
            if charcode in non_printing:
                self.paint_non_printing(painter, option, charcode)
            else:
                self.paint_normal(painter, option, charcode)
        finally:
            painter.restore()

    def paint_normal(self, painter, option, charcode):
        if False:
            for i in range(10):
                print('nop')
        f = option.font
        f.setPixelSize(option.rect.height() - 8)
        painter.setFont(f)
        painter.drawText(option.rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom | Qt.TextFlag.TextSingleLine, codepoint_to_chr(charcode))

    def paint_non_printing(self, painter, option, charcode):
        if False:
            return 10
        text = self.np_pat.sub('\\n\\1', non_printing[charcode])
        painter.drawText(option.rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap | Qt.TextFlag.TextWrapAnywhere, text)
        painter.setPen(QPen(Qt.PenStyle.DashLine))
        painter.drawRect(option.rect.adjusted(1, 1, -1, -1))

class CharView(QListView):
    show_name = pyqtSignal(object)
    char_selected = pyqtSignal(object)

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        self.last_mouse_idx = -1
        QListView.__init__(self, parent)
        self._model = CharModel(self)
        self.setModel(self._model)
        self.delegate = CharDelegate(self)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setItemDelegate(self.delegate)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setWrapping(True)
        self.setMouseTracking(True)
        self.setSpacing(2)
        self.setUniformItemSizes(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.context_menu)
        self.showing_favorites = False
        set_no_activate_on_click(self)
        self.activated.connect(self.item_activated)
        self.clicked.connect(self.item_activated)

    def item_activated(self, index):
        if False:
            for i in range(10):
                print('nop')
        try:
            char_code = int(self.model().data(index, Qt.ItemDataRole.UserRole))
        except (TypeError, ValueError):
            pass
        else:
            self.char_selected.emit(codepoint_to_chr(char_code))

    def set_allow_drag_and_drop(self, enabled):
        if False:
            for i in range(10):
                print('nop')
        if not enabled:
            self.setDragEnabled(False)
            self.viewport().setAcceptDrops(False)
            self.setDropIndicatorShown(True)
            self._model.allow_dnd = False
        else:
            self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            self.viewport().setAcceptDrops(True)
            self.setDragEnabled(True)
            self.setAcceptDrops(True)
            self.setDropIndicatorShown(False)
            self._model.allow_dnd = True

    def show_chars(self, name, codes):
        if False:
            while True:
                i = 10
        self.showing_favorites = name == _('Favorites')
        self._model.beginResetModel()
        self._model.chars = codes
        self._model.endResetModel()
        self.scrollToTop()

    def mouseMoveEvent(self, ev):
        if False:
            i = 10
            return i + 15
        index = self.indexAt(ev.pos())
        if index.isValid():
            row = index.row()
            if row != self.last_mouse_idx:
                self.last_mouse_idx = row
                try:
                    char_code = int(self.model().data(index, Qt.ItemDataRole.UserRole))
                except (TypeError, ValueError):
                    pass
                else:
                    self.show_name.emit(char_code)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.show_name.emit(-1)
            self.last_mouse_idx = -1
        return QListView.mouseMoveEvent(self, ev)

    def context_menu(self, pos):
        if False:
            return 10
        index = self.indexAt(pos)
        if index.isValid():
            try:
                char_code = int(self.model().data(index, Qt.ItemDataRole.UserRole))
            except (TypeError, ValueError):
                pass
            else:
                m = QMenu(self)
                m.addAction(QIcon.ic('edit-copy.png'), _('Copy %s to clipboard') % codepoint_to_chr(char_code), partial(self.copy_to_clipboard, char_code))
                m.addAction(QIcon.ic('rating.png'), (_('Remove %s from favorites') if self.showing_favorites else _('Add %s to favorites')) % codepoint_to_chr(char_code), partial(self.remove_from_favorites, char_code))
                if self.showing_favorites:
                    m.addAction(_('Restore favorites to defaults'), self.restore_defaults)
                m.exec(self.mapToGlobal(pos))

    def restore_defaults(self):
        if False:
            return 10
        del tprefs['charmap_favorites']
        self.model().beginResetModel()
        self.model().chars = list(tprefs['charmap_favorites'])
        self.model().endResetModel()

    def copy_to_clipboard(self, char_code):
        if False:
            return 10
        c = QApplication.clipboard()
        c.setText(codepoint_to_chr(char_code))

    def remove_from_favorites(self, char_code):
        if False:
            i = 10
            return i + 15
        existing = tprefs['charmap_favorites']
        if not self.showing_favorites:
            if char_code not in existing:
                tprefs['charmap_favorites'] = [char_code] + existing
        elif char_code in existing:
            existing.remove(char_code)
            tprefs['charmap_favorites'] = existing
            self.model().beginResetModel()
            self.model().chars.remove(char_code)
            self.model().endResetModel()

class CharSelect(Dialog):

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        self.initialized = False
        Dialog.__init__(self, _('Insert character'), 'charmap_dialog', parent)
        self.setWindowIcon(QIcon.ic('character-set.png'))
        self.setFocusProxy(parent)

    def setup_ui(self):
        if False:
            while True:
                i = 10
        self.l = l = QGridLayout(self)
        self.setLayout(l)
        self.bb.setStandardButtons(QDialogButtonBox.StandardButton.Close)
        self.rearrange_button = b = self.bb.addButton(_('Re-arrange favorites'), QDialogButtonBox.ButtonRole.ActionRole)
        b.setCheckable(True)
        b.setChecked(False)
        b.setVisible(False)
        b.setDefault(True)
        self.splitter = s = QSplitter(self)
        s.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        s.setChildrenCollapsible(False)
        self.search = h = HistoryLineEdit2(self)
        h.setToolTip(textwrap.fill(_('Search for Unicode characters by using the English names or nicknames. You can also search directly using a character code. For example, the following searches will all yield the no-break space character: U+A0, nbsp, no-break')))
        h.initialize('charmap_search')
        h.setPlaceholderText(_('Search by name, nickname or character code'))
        self.search_button = b = QPushButton(_('&Search'))
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        h.returnPressed.connect(self.do_search)
        b.clicked.connect(self.do_search)
        self.clear_button = cb = QToolButton(self)
        cb.setIcon(QIcon.ic('clear_left.png'))
        cb.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        cb.setText(_('Clear search'))
        cb.clicked.connect(self.clear_search)
        (l.addWidget(h), l.addWidget(b, 0, 1), l.addWidget(cb, 0, 2))
        self.category_view = CategoryView(self)
        self.category_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.category_view.clicked.connect(self.category_view_clicked)
        l.addWidget(s, 1, 0, 1, 3)
        self.char_view = CharView(self)
        self.char_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.rearrange_button.toggled[bool].connect(self.set_allow_drag_and_drop)
        self.category_view.category_selected.connect(self.show_chars)
        self.char_view.show_name.connect(self.show_char_info)
        self.char_view.char_selected.connect(self.char_selected)
        (s.addWidget(self.category_view), s.addWidget(self.char_view))
        self.char_info = la = QLabel('\xa0')
        la.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        l.addWidget(la, 2, 0, 1, 3)
        self.rearrange_msg = la = QLabel(_('Drag and drop characters to re-arrange them. Click the "Re-arrange" button again when you are done.'))
        la.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        la.setVisible(False)
        l.addWidget(la, 3, 0, 1, 3)
        self.h = h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        self.match_any = mm = QCheckBox(_('Match any word'))
        mm.setToolTip(_('When searching return characters whose names match any of the specified words'))
        mm.setChecked(tprefs.get('char_select_match_any', True))
        connect_lambda(mm.stateChanged, self, lambda self: tprefs.set('char_select_match_any', self.match_any.isChecked()))
        (h.addWidget(mm), h.addStretch(), h.addWidget(self.bb))
        l.addLayout(h, 4, 0, 1, 3)
        self.char_view.setFocus(Qt.FocusReason.OtherFocusReason)

    def category_view_clicked(self):
        if False:
            return 10
        p = self.parent()
        if p is not None and p.focusWidget() is not None:
            p.activateWindow()

    def do_search(self):
        if False:
            while True:
                i = 10
        text = str(self.search.text()).strip()
        if not text:
            return self.clear_search()
        with BusyCursor():
            chars = search_for_chars(text, and_tokens=not self.match_any.isChecked())
        self.show_chars(_('Search'), chars)

    def clear_search(self):
        if False:
            for i in range(10):
                print('nop')
        self.search.clear()
        self.category_view.get_chars()

    def set_allow_drag_and_drop(self, on):
        if False:
            return 10
        self.char_view.set_allow_drag_and_drop(on)
        self.rearrange_msg.setVisible(on)

    def show_chars(self, name, codes):
        if False:
            for i in range(10):
                print('nop')
        b = self.rearrange_button
        b.setVisible(name == _('Favorites'))
        b.blockSignals(True)
        b.setChecked(False)
        b.blockSignals(False)
        self.char_view.show_chars(name, codes)
        self.char_view.set_allow_drag_and_drop(False)

    def initialize(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.initialized:
            self.category_view.initialize()

    def sizeHint(self):
        if False:
            i = 10
            return i + 15
        return QSize(800, 600)

    def show_char_info(self, char_code):
        if False:
            return 10
        text = '\xa0'
        if char_code > 0:
            (category_name, subcategory_name, character_name) = self.category_view.model().get_char_info(char_code)
            text = _('{character_name} (U+{char_code:04X}) in {category_name} - {subcategory_name}').format(**locals())
        self.char_info.setText(text)

    def show(self):
        if False:
            return 10
        self.initialize()
        Dialog.show(self)
        self.raise_()

    def char_selected(self, c):
        if False:
            print('Hello World!')
        if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier:
            self.hide()
        if self.parent() is None or self.parent().focusWidget() is None:
            QApplication.clipboard().setText(c)
            return
        self.parent().activateWindow()
        w = self.parent().focusWidget()
        e = QInputMethodEvent('', [])
        e.setCommitString(c)
        if hasattr(w, 'no_popup'):
            oval = w.no_popup
            w.no_popup = True
        QApplication.sendEvent(w, e)
        if hasattr(w, 'no_popup'):
            w.no_popup = oval
if __name__ == '__main__':
    from calibre.gui2 import Application
    app = Application([])
    w = CharSelect()
    w.initialize()
    w.show()
    app.exec()