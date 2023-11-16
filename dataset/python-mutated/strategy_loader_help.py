import sys
import traceback
import six
from rqalpha.utils.exception import patch_user_exc, CustomError, CustomException

def compile_strategy(source_code, strategy, scope):
    if False:
        for i in range(10):
            print('nop')
    try:
        code = compile(source_code, strategy, 'exec')
        six.exec_(code, scope)
        return scope
    except Exception as e:
        (exc_type, exc_val, exc_tb) = sys.exc_info()
        exc_val = patch_user_exc(exc_val, force=True)
        try:
            msg = str(exc_val)
        except Exception as e1:
            msg = ''
            six.print_(e1)
        error = CustomError()
        error.set_msg(msg)
        error.set_exc(exc_type, exc_val, exc_tb)
        stackinfos = list(traceback.extract_tb(exc_tb))
        if isinstance(e, (SyntaxError, IndentationError)):
            error.add_stack_info(exc_val.filename, exc_val.lineno, '', exc_val.text)
        else:
            for item in stackinfos:
                (filename, lineno, func_name, code) = item
                if strategy == filename:
                    error.add_stack_info(*item)
            if error.stacks_length == 0:
                error.add_stack_info(*item)
        raise CustomException(error)