"""Code transformation exceptions."""
import collections
from nvidia.dali._autograph.pyct import origin_info

class FrameInfo(collections.namedtuple('FrameInfo', ('filename', 'lineno', 'function_name', 'code', 'is_converted', 'is_allowlisted'))):
    __slots__ = ()

def _stack_trace_inside_mapped_code(tb, source_map, converter_filename):
    if False:
        while True:
            i = 10
    'Summarizes inner traceback frames up to the call to a given function.\n\n  This functions locates the innermost (i.e. most recent) frame that corresponds\n  to code that can be mapped by source_map originated from, and returns a\n  translated stack trace ending at that frame. If no such frame is found, the\n  entire stack trace is summarized.\n\n  For example, the following code:\n\n    def f():\n      for i in tf.range(1):\n        z = y + i  # z only defined here\n\n  Would generate this traceback:\n\n    <converted code>\n        ag__.for_stmt(...)\n    <for_stmt>\n        return _known_len_tf_for_stmt(iter_, extra_test, body, init_state)\n    <_known_len_tf_for_stmt>\n        _disallow_undefs_into_loop(*init_state)\n    <_disallow_undefs_into_loop>\n        raise ...\n\n  Which is then processed into:\n\n    <f>\n        for i in tf.range(1):\n    <for_stmt>\n        return _known_len_tf_for_stmt(iter_, extra_test, body, init_state)\n    <_known_len_tf_for_stmt>\n        _disallow_undefs_into_loop(*init_state)\n    <_disallow_undefs_into_loop>\n        raise ...\n\n  Args:\n    tb: traceback.FrameSummary, The traceback corresponding to an error.\n      Typically, the output of traceback.Summary.extract(capture_locals=True).\n    source_map: Dict[LineLocation, OriginInfo], a source map as created by\n      origin_info.create_source_map.\n    converter_filename: str, the file path of the converted module. Call frames\n      corresponding to this module are elided and their preceding frames are\n      marked as allowlisted. Note that frames enclosing converted code are\n      dropped using a different mechanism.\n\n  Returns:\n    List[FrameInfo]\n  '
    result_frames = []
    for (filename, line_number, function_name, text) in reversed(tb):
        loc = origin_info.LineLocation(filename=filename, lineno=line_number)
        if loc in source_map:
            origin = source_map[loc]
            fi = FrameInfo(filename=origin.loc.filename, lineno=origin.loc.lineno, function_name=origin.function_name, code=origin.source_code_line, is_converted=True, is_allowlisted=False)
            result_frames.append(fi)
            break
        if filename == converter_filename:
            if result_frames:
                prev = result_frames[-1]
                assert not prev.is_converted
                fi = FrameInfo(filename=prev.filename, lineno=prev.lineno, function_name=prev.function_name, code=prev.code, is_converted=False, is_allowlisted=True)
                result_frames[-1] = fi
            continue
        fi = FrameInfo(filename=filename, lineno=line_number, function_name=function_name, code=text, is_converted=False, is_allowlisted=False)
        result_frames.append(fi)
    return tuple(result_frames)
KNOWN_STRING_CONSTRUCTOR_ERRORS = (AssertionError, AttributeError, NameError, NotImplementedError, RuntimeError, StopIteration, TypeError, UnboundLocalError, ValueError)

class MultilineMessageKeyError(KeyError):

    def __init__(self, message, original_key):
        if False:
            while True:
                i = 10
        super(MultilineMessageKeyError, self).__init__(original_key)
        self.__message = message

    def __str__(self):
        if False:
            print('Hello World!')
        return self.__message
MultilineMessageKeyError.__name__ = KeyError.__name__

class ErrorMetadataBase(object):
    """Container objects attached to exceptions raised in user code.

  This metadata allows re-raising exceptions that occur in generated code, with
  a custom error message that includes a stack trace relative to user-readable
  code from which the generated code originated.
  """
    __slots__ = ('translated_stack', 'cause_message')

    def __init__(self, callsite_tb, cause_metadata, cause_message, source_map, converter_filename):
        if False:
            print('Hello World!')
        translated_stack = _stack_trace_inside_mapped_code(callsite_tb, source_map, converter_filename)
        if cause_metadata is None:
            self.translated_stack = translated_stack
            self.cause_message = cause_message
        else:
            self.translated_stack = cause_metadata.translated_stack + (translated_stack[-1],)
            self.cause_message = cause_metadata.cause_message

    def get_message(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the message for the underlying exception.'
        lines = []
        lines.append('in user code:')
        lines.append('')
        for frame_info in reversed(self.translated_stack):
            formatted_line = f'    File "{frame_info.filename}", line {frame_info.lineno}, in {frame_info.function_name}'
            if frame_info.is_converted:
                formatted_line += '  *'
            elif frame_info.is_allowlisted:
                formatted_line += '  **'
            lines.append(formatted_line)
            if frame_info.code is None:
                code_snippet = '<source unavailable>'
            else:
                code_snippet = frame_info.code.strip()
            lines.append('        {}'.format(code_snippet))
        lines.append('')
        message_lines = self.cause_message.split('\n')
        for i in range(len(message_lines)):
            message_lines[i] = '    ' + message_lines[i]
        lines.extend(message_lines)
        lines.append('')
        return '\n'.join(lines)

    def create_exception(self, source_error):
        if False:
            while True:
                i = 10
        'Creates exception from source_error.'
        preferred_type = type(source_error)
        to_ret = None
        if preferred_type.__init__ is Exception.__init__:
            to_ret = preferred_type(self.get_message())
        if preferred_type in KNOWN_STRING_CONSTRUCTOR_ERRORS:
            to_ret = preferred_type(self.get_message())
        elif preferred_type is KeyError:
            to_ret = MultilineMessageKeyError(self.get_message(), self.cause_message)
        if to_ret is not None:
            return to_ret.with_traceback(source_error.__traceback__)

    def to_exception(self, source_error):
        if False:
            while True:
                i = 10
        exc = self.create_exception(source_error)
        exc.__suppress_context__ = True
        exc.ag_error_metadata = self
        return exc