import inspect
import logging
import pprint
import string
from qtpy.QtCore import Qt
from qtpy.QtGui import QTextOption
from spyder_kernels.utils.lazymodules import numpy as np
from spyder_kernels.utils.nsview import get_size, get_human_readable_type, value_to_display
from spyder.config.base import _
from spyder.py3compat import TEXT_TYPES, to_text_string
SMALL_COL_WIDTH = 120
MEDIUM_COL_WIDTH = 200
_PRETTY_PRINTER = pprint.PrettyPrinter(indent=4)
_ALL_PREDICATES = (inspect.ismodule, inspect.isclass, inspect.ismethod, inspect.isfunction, inspect.isgeneratorfunction, inspect.isgenerator, inspect.istraceback, inspect.isframe, inspect.iscode, inspect.isbuiltin, inspect.isroutine, inspect.isabstract, inspect.ismethoddescriptor, inspect.isdatadescriptor, inspect.isgetsetdescriptor, inspect.ismemberdescriptor)
ALIGN_LEFT = int(Qt.AlignVCenter | Qt.AlignLeft)
ALIGN_RIGHT = int(Qt.AlignVCenter | Qt.AlignRight)
logger = logging.getLogger(__name__)

def tio_call(obj_fn, tree_item):
    if False:
        i = 10
        return i + 15
    'Calls obj_fn(tree_item.obj).'
    return obj_fn(tree_item.obj)

def safe_tio_call(obj_fn, tree_item, log_exceptions=False):
    if False:
        while True:
            i = 10
    '\n    Call the obj_fn(tree_item.obj).\n    Returns empty string in case of an error.\n    '
    tio = tree_item.obj
    try:
        return str(obj_fn(tio))
    except Exception as ex:
        if log_exceptions:
            logger.exception(ex)
        return ''

def safe_data_fn(obj_fn, log_exceptions=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates a function that returns an empty string in case of an exception.\n\n    :param fnobj_fn: function that will be wrapped\n    :type obj_fn: object to basestring function\n    :returns: function that can be used as AttributeModel data_fn attribute\n    :rtype: objbrowser.treeitem.TreeItem to string function\n    '

    def data_fn(tree_item):
        if False:
            return 10
        '\n        Call the obj_fn(tree_item.obj).\n        Returns empty string in case of an error.\n        '
        return safe_tio_call(obj_fn, tree_item, log_exceptions=log_exceptions)
    return data_fn

def tio_predicates(tree_item):
    if False:
        print('Hello World!')
    'Returns the inspect module predicates that are true for this object.'
    tio = tree_item.obj
    predicates = [pred.__name__ for pred in _ALL_PREDICATES if pred(tio)]
    return ', '.join(predicates)

def tio_summary(tree_item):
    if False:
        i = 10
        return i + 15
    '\n    Returns a small summary of regular objects.\n    For callables and modules an empty string is returned.\n    '
    tio = tree_item.obj
    if isinstance(tio, TEXT_TYPES):
        return tio
    elif isinstance(tio, (list, tuple, set, frozenset, dict)):
        n_items = len(tio)
        if n_items == 0:
            return _('empty {}').format(type(tio).__name__)
        if n_items == 1:
            return _('{} of {} item').format(type(tio).__name__, n_items)
        else:
            return _('{} of {} items').format(type(tio).__name__, n_items)
    elif isinstance(tio, np.ndarray):
        return _('array of {}, shape: {}').format(tio.dtype, tio.shape)
    elif callable(tio) or inspect.ismodule(tio):
        return ''
    else:
        return str(tio)

def tio_is_attribute(tree_item):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns 'True' if the tree item object is an attribute of the parent\n    opposed to e.g. a list element.\n    "
    if tree_item.is_attribute is None:
        return ''
    else:
        return str(tree_item.is_attribute)

def tio_is_callable(tree_item):
    if False:
        for i in range(10):
            print('nop')
    "Returns 'True' if the tree item object is callable."
    return str(callable(tree_item.obj))

def tio_doc_str(tree_item):
    if False:
        return 10
    'Returns the doc string of an object.'
    tio = tree_item.obj
    try:
        return tio.__doc__
    except AttributeError:
        return _('<no doc string found>')

class AttributeModel(object):
    """
    Determines how an object attribute is rendered
    in a table column or details pane.
    """

    def __init__(self, name, doc=_('<no help available>'), data_fn=None, col_visible=True, width=SMALL_COL_WIDTH, alignment=ALIGN_LEFT, line_wrap=QTextOption.NoWrap):
        if False:
            return 10
        '\n        Constructor\n\n        :param name: name used to describe the attribute\n        :type name: string\n        :param doc: short string documenting the attribute\n        :type doc: string\n        :param data_fn: function that calculates the value shown in the UI\n        :type  data_fn: function(TreeItem_ to string.\n        :param col_visible: if True, the attribute is col_visible by default\n                            in the table\n        :type col_visible: bool\n        :param width: default width in the attribute table\n        :type with: int\n        :param alignment: alignment of the value in the table\n        :type alignment: Qt.AlignmentFlag\n        :param line_wrap: Line wrap mode of the attribute in the details pane\n        :type line_wrap: QtGui.QPlainTextEdit\n        '
        if not callable(data_fn):
            raise ValueError('data_fn must be function(TreeItem)->string')
        self.name = name
        self.doc = doc
        self.data_fn = data_fn
        self.col_visible = col_visible
        self.width = width
        self.alignment = alignment
        self.line_wrap = line_wrap

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'String representation.'
        return _('<AttributeModel for {!r}>').format(self.name)

    @property
    def settings_name(self):
        if False:
            i = 10
            return i + 15
        'The name where spaces are replaced by underscores.'
        sname = self.name.replace(' ', '_')
        return sname.translate(None, string.punctuation).translate(None, string.whitespace)
ATTR_MODEL_VALUE = AttributeModel('Value', doc=_('The value of the object.'), data_fn=lambda tree_item: value_to_display(tree_item.obj), col_visible=True, width=SMALL_COL_WIDTH)
ATTR_MODEL_NAME = AttributeModel('Name', doc=_('The name of the object.'), data_fn=lambda tree_item: tree_item.obj_name if tree_item.obj_name else _('<root>'), col_visible=True, width=SMALL_COL_WIDTH)
ATTR_MODEL_PATH = AttributeModel('Path', doc=_("A path to the data: e.g. var[1]['a'].item"), data_fn=lambda tree_item: tree_item.obj_path if tree_item.obj_path else _('<root>'), col_visible=True, width=MEDIUM_COL_WIDTH)
ATTR_MODEL_SUMMARY = AttributeModel('summary', doc=_('A summary of the object for regular objects (is empty for non-regular objectssuch as callables or modules).'), data_fn=tio_summary, col_visible=True, alignment=ALIGN_LEFT, width=MEDIUM_COL_WIDTH)
ATTR_MODEL_UNICODE = AttributeModel('unicode', doc=_('The unicode representation of the object. In Python 2 it uses unicode()In Python 3 the str() function is used.'), data_fn=lambda tree_item: to_text_string(tree_item.obj), col_visible=False, width=MEDIUM_COL_WIDTH, line_wrap=QTextOption.WrapAtWordBoundaryOrAnywhere)
ATTR_MODEL_STR = AttributeModel('str', doc=_("The string representation of the object using the str() function.In Python 3 there is no difference with the 'unicode' column."), data_fn=lambda tree_item: str(tree_item.obj), col_visible=False, width=MEDIUM_COL_WIDTH, line_wrap=QTextOption.WrapAtWordBoundaryOrAnywhere)
ATTR_MODEL_REPR = AttributeModel('repr', doc=_('The string representation of the object using the repr() function.'), data_fn=lambda tree_item: repr(tree_item.obj), col_visible=True, width=MEDIUM_COL_WIDTH, line_wrap=QTextOption.WrapAtWordBoundaryOrAnywhere)
ATTR_MODEL_TYPE = AttributeModel('type function', doc=_('Type of the object determined using the builtin type() function'), data_fn=lambda tree_item: str(type(tree_item.obj)), col_visible=False, width=MEDIUM_COL_WIDTH)
ATTR_MODEL_CLASS = AttributeModel('Type', doc='The name of the class of the object via obj.__class__.__name__', data_fn=lambda tree_item: get_human_readable_type(tree_item.obj), col_visible=True, width=MEDIUM_COL_WIDTH)
ATTR_MODEL_LENGTH = AttributeModel('Size', doc=_('The length or shape of the object'), data_fn=lambda tree_item: to_text_string(get_size(tree_item.obj)), col_visible=True, alignment=ALIGN_RIGHT, width=SMALL_COL_WIDTH)
ATTR_MODEL_ID = AttributeModel('Id', doc=_('The identifier of the object with calculated using the id() function'), data_fn=lambda tree_item: '0x{:X}'.format(id(tree_item.obj)), col_visible=False, alignment=ALIGN_RIGHT, width=SMALL_COL_WIDTH)
ATTR_MODEL_IS_ATTRIBUTE = AttributeModel('Attribute', doc=_('The object is an attribute of the parent (opposed to e.g. a list element).Attributes are displayed in italics in the table.'), data_fn=tio_is_attribute, col_visible=False, width=SMALL_COL_WIDTH)
ATTR_MODEL_CALLABLE = AttributeModel('Callable', doc=_('True if the object is callable.Determined with the `callable` built-in function.Callable objects are displayed in blue in the table.'), data_fn=tio_is_callable, col_visible=True, width=SMALL_COL_WIDTH)
ATTR_MODEL_IS_ROUTINE = AttributeModel('Routine', doc=_('True if the object is a user-defined or built-in function or method.Determined with the inspect.isroutine() method.'), data_fn=lambda tree_item: str(inspect.isroutine(tree_item.obj)), col_visible=False, width=SMALL_COL_WIDTH)
ATTR_MODEL_PRED = AttributeModel('inspect predicates', doc=_('Predicates from the inspect module'), data_fn=tio_predicates, col_visible=False, width=MEDIUM_COL_WIDTH)
ATTR_MODEL_PRETTY_PRINT = AttributeModel('pretty print', doc=_('Pretty printed representation of the object using the pprint module.'), data_fn=lambda tree_item: _PRETTY_PRINTER.pformat(tree_item.obj), col_visible=False, width=MEDIUM_COL_WIDTH)
ATTR_MODEL_DOC_STRING = AttributeModel('doc string', doc=_("The object's doc string"), data_fn=tio_doc_str, col_visible=False, width=MEDIUM_COL_WIDTH)
ATTR_MODEL_GET_DOC = AttributeModel('Documentation', doc=_("The object's doc string, leaned up by inspect.getdoc()"), data_fn=safe_data_fn(inspect.getdoc), col_visible=False, width=MEDIUM_COL_WIDTH)
ATTR_MODEL_GET_COMMENTS = AttributeModel('inspect.getcomments', doc=_("Comments above the object's definition. Retrieved using inspect.getcomments()"), data_fn=lambda tree_item: inspect.getcomments(tree_item.obj), col_visible=False, width=MEDIUM_COL_WIDTH)
ATTR_MODEL_GET_MODULE = AttributeModel('inspect.getmodule', doc=_("The object's module. Retrieved using inspect.module"), data_fn=safe_data_fn(inspect.getmodule), col_visible=False, width=MEDIUM_COL_WIDTH)
ATTR_MODEL_GET_FILE = AttributeModel('File', doc=_("The object's file. Retrieved using inspect.getfile"), data_fn=safe_data_fn(inspect.getfile), col_visible=False, width=MEDIUM_COL_WIDTH)
ATTR_MODEL_GET_SOURCE_FILE = AttributeModel('Source file', doc=_("The object's file. Retrieved using inspect.getsourcefile"), data_fn=safe_data_fn(inspect.getsourcefile), col_visible=False, width=MEDIUM_COL_WIDTH)
ATTR_MODEL_GET_SOURCE_LINES = AttributeModel('inspect.getsourcelines', doc=_('Uses inspect.getsourcelines() to get a list of source lines for the object'), data_fn=safe_data_fn(inspect.getsourcelines), col_visible=False, width=MEDIUM_COL_WIDTH)
ATTR_MODEL_GET_SOURCE = AttributeModel('Source code', doc=_('The source code of an object retrieved using inspect.getsource'), data_fn=safe_data_fn(inspect.getsource), col_visible=False, width=MEDIUM_COL_WIDTH)
ALL_ATTR_MODELS = (ATTR_MODEL_NAME, ATTR_MODEL_PATH, ATTR_MODEL_SUMMARY, ATTR_MODEL_UNICODE, ATTR_MODEL_STR, ATTR_MODEL_REPR, ATTR_MODEL_TYPE, ATTR_MODEL_CLASS, ATTR_MODEL_LENGTH, ATTR_MODEL_ID, ATTR_MODEL_IS_ATTRIBUTE, ATTR_MODEL_CALLABLE, ATTR_MODEL_IS_ROUTINE, ATTR_MODEL_PRED, ATTR_MODEL_PRETTY_PRINT, ATTR_MODEL_DOC_STRING, ATTR_MODEL_GET_DOC, ATTR_MODEL_GET_COMMENTS, ATTR_MODEL_GET_MODULE, ATTR_MODEL_GET_FILE, ATTR_MODEL_GET_SOURCE_FILE, ATTR_MODEL_GET_SOURCE_LINES, ATTR_MODEL_GET_SOURCE)
DEFAULT_ATTR_COLS = (ATTR_MODEL_NAME, ATTR_MODEL_CLASS, ATTR_MODEL_LENGTH, ATTR_MODEL_VALUE, ATTR_MODEL_CALLABLE, ATTR_MODEL_PATH, ATTR_MODEL_ID, ATTR_MODEL_IS_ATTRIBUTE, ATTR_MODEL_IS_ROUTINE, ATTR_MODEL_GET_FILE, ATTR_MODEL_GET_SOURCE_FILE)
DEFAULT_ATTR_DETAILS = (ATTR_MODEL_GET_DOC, ATTR_MODEL_GET_SOURCE, ATTR_MODEL_GET_FILE)
assert len(ALL_ATTR_MODELS) == len(set(ALL_ATTR_MODELS))
assert len(DEFAULT_ATTR_COLS) == len(set(DEFAULT_ATTR_COLS))
assert len(DEFAULT_ATTR_DETAILS) == len(set(DEFAULT_ATTR_DETAILS))