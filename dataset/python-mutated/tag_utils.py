import pmt
from . import gr_python as gr

class PythonTag(object):
    """ Python container for tags """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.offset = None
        self.key = None
        self.value = None
        self.srcid = False

def tag_to_python(tag):
    if False:
        i = 10
        return i + 15
    ' Convert a stream tag to a Python-readable object '
    newtag = PythonTag()
    newtag.offset = tag.offset
    newtag.key = pmt.to_python(tag.key)
    newtag.value = pmt.to_python(tag.value)
    newtag.srcid = pmt.to_python(tag.srcid)
    return newtag

def python_to_tag(tag_struct):
    if False:
        i = 10
        return i + 15
    "\n    Convert a Python list/tuple/dictionary to a stream tag.\n\n    When using a list or tuple format, this function expects the format:\n      tag_struct[0] --> tag's offset (as an integer)\n      tag_struct[1] --> tag's key (as a PMT)\n      tag_struct[2] --> tag's value (as a PMT)\n      tag_struct[3] --> tag's srcid (as a PMT)\n\n    When using a dictionary, we specify the dictionary keys using:\n      tag_struct['offset'] --> tag's offset (as an integer)\n      tag_struct['key'] --> tag's key (as a PMT)\n      tag_struct['value'] --> tag's value (as a PMT)\n      tag_struct['srcid'] --> tag's srcid (as a PMT)\n\n    If the function can take the Python object and successfully\n    construct a tag, it will return the tag. Otherwise, it will return\n    None.\n    "
    good = False
    tag = gr.tag_t()
    if type(tag_struct) == dict:
        if 'offset' in tag_struct:
            if isinstance(tag_struct['offset'], int):
                tag.offset = tag_struct['offset']
                good = True
        if 'key' in tag_struct:
            if isinstance(tag_struct['key'], pmt.pmt_base):
                tag.key = tag_struct['key']
                good = True
        if 'value' in tag_struct:
            if isinstance(tag_struct['value'], pmt.pmt_base):
                tag.value = tag_struct['value']
                good = True
        if 'srcid' in tag_struct:
            if isinstance(tag_struct['srcid'], pmt.pmt_base):
                tag.srcid = tag_struct['srcid']
                good = True
    elif type(tag_struct) == list or type(tag_struct) == tuple:
        if len(tag_struct) == 4:
            if isinstance(tag_struct[0], int):
                tag.offset = tag_struct[0]
                good = True
            if isinstance(tag_struct[1], pmt.pmt_base):
                tag.key = tag_struct[1]
                good = True
            if isinstance(tag_struct[2], pmt.pmt_base):
                tag.value = tag_struct[2]
                good = True
            if isinstance(tag_struct[3], pmt.pmt_base):
                tag.srcid = tag_struct[3]
                good = True
        elif len(tag_struct) == 3:
            if isinstance(tag_struct[0], int):
                tag.offset = tag_struct[0]
                good = True
            if isinstance(tag_struct[1], pmt.pmt_base):
                tag.key = tag_struct[1]
                good = True
            if isinstance(tag_struct[2], pmt.pmt_base):
                tag.value = tag_struct[2]
                good = True
            tag.srcid = pmt.PMT_F
    if good:
        return tag
    else:
        return None