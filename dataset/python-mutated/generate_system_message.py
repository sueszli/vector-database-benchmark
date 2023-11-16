import traceback
from ..rag.get_relevant_procedures_string import get_relevant_procedures_string
from ..utils.get_user_info_string import get_user_info_string

def generate_system_message(interpreter):
    if False:
        print('Hello World!')
    '\n    Dynamically generate a system message.\n\n    Takes an interpreter instance,\n    returns a string.\n\n    This is easy to replace!\n    Just swap out `interpreter.generate_system_message` with another function.\n    '
    system_message = interpreter.system_message
    system_message += '\n' + get_user_info_string()
    if not interpreter.local and (not interpreter.disable_procedures):
        try:
            system_message += '\n' + get_relevant_procedures_string(interpreter.messages)
        except:
            if interpreter.debug_mode:
                print(traceback.format_exc())
    return system_message